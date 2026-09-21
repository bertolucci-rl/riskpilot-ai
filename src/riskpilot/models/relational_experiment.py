"""Milestone 3 experiment: how much do the relational (historical) tables add?

Run from the command line::

    python -m riskpilot.models.relational_experiment                    # all stages
    python -m riskpilot.models.relational_experiment --stage ablation   # source ablation only
    python -m riskpilot.models.relational_experiment --stage final --resume

Protocol (the report explains the reasoning):

* The Milestone 1 split is reloaded and verified; the Milestone 2 internal
  validation split (stratified holdout carved out of the frozen training
  portion, seed 42) is reconstructed and persisted. The frozen **test**
  partition is touched only by :func:`run_final_stage`.
* **Ablation** (validation only, frozen LightGBM configuration from Milestone 2,
  fixed rounds, no retuning): application-only ``M0``, the cumulative sequence
  ``M1 .. M5``, single-source models ``S_<source>`` and leave-one-out models
  ``L-<source>``. Sources are then retained by backward elimination: the source
  whose removal costs the least validation log loss is dropped when the paired
  bootstrap interval of that cost includes zero, and the elimination stops at
  the first source whose removal reliably hurts.
* **Retuning** (optional, small): a coordinate search over the parameters that
  a wider feature space plausibly affects, with early stopping on validation,
  on the retained feature set only.
* **Final**: the locked feature set and configuration are refitted on the whole
  training portion and scored once on the frozen test split, next to the
  persisted Milestone 1 Logistic Regression and Milestone 2 application-only
  LightGBM (both reloaded and checked against their recorded metrics), with a
  paired bootstrap, calibration diagnostics and tree-based importance by source.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from riskpilot import __version__, config
from riskpilot.features.relational import assemble
from riskpilot.models import challengers as ch
from riskpilot.models import comparison
from riskpilot.models.evaluate import (
    MODEL_COLOR,
    REFERENCE_COLOR,
    SECOND_COLOR,
    _finish,
    _style_axes,
    calibration_table,
    compute_metrics,
    make_evaluation_figures,
    save_metrics,
    use_headless_backend,
)
from riskpilot.models.train import _display_path, make_split

logger = logging.getLogger(__name__)

MODEL_KEY = "lightgbm_relational"
APP_ONLY_KEY = "lightgbm"
SOURCES: tuple[str, ...] = assemble.SOURCES
SEQUENCE: tuple[str, ...] = ("bureau", "previous", "installments", "credit_card", "pos")
RETUNE_AXES: list[tuple[str, list[Any]]] = [
    ("num_leaves", [15, 31]),
    ("min_child_samples", [100, 500]),
    ("colsample_bytree", [0.3, 0.5]),
    ("reg_lambda", [1.0, 10.0]),
]
SMOKE_RETUNE_AXES: list[tuple[str, list[Any]]] = [("num_leaves", [7, 15])]
METRIC_KEYS = (
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "brier_skill_score",
    "expected_calibration_error",
    "calibration_slope",
    "calibration_intercept",
)
GROUP_COLORS = {"sequential": MODEL_COLOR, "single": SECOND_COLOR, "leave-one-out": "#1baf7a"}

__all__ = [
    "RETUNE_AXES",
    "SEQUENCE",
    "AblationConfig",
    "RelationalPaths",
    "ablation_configs",
    "backward_elimination",
    "columns_for_sources",
    "importance_by_source",
    "locked_lightgbm_spec",
    "main",
    "run_ablation_stage",
    "run_final_stage",
    "run_retune_stage",
]


# --------------------------------------------------------------------------- #
# Paths and configuration
# --------------------------------------------------------------------------- #
@dataclass
class RelationalPaths:
    data_path: Path = config.APPLICATION_TRAIN_PATH
    processed_dir: Path = config.PROCESSED_DATA_DIR
    relational_dir: Path = config.RELATIONAL_PROCESSED_DIR
    metrics_dir: Path = config.METRICS_DIR
    figures_dir: Path = config.FIGURES_DIR
    models_dir: Path = config.MODELS_DIR

    @property
    def membership_path(self) -> Path:
        return self.processed_dir / "baseline_split_membership.csv"

    @property
    def internal_membership_path(self) -> Path:
        return self.processed_dir / "internal_validation_membership.csv"

    @property
    def validation_predictions_path(self) -> Path:
        return self.processed_dir / "relational_validation_predictions.parquet"

    @property
    def test_predictions_path(self) -> Path:
        return self.processed_dir / "relational_test_predictions.csv"

    @property
    def challenger_selection_path(self) -> Path:
        return self.metrics_dir / "challenger_selection.json"

    @property
    def baseline_metrics_path(self) -> Path:
        return self.metrics_dir / "baseline_metrics.json"

    @property
    def baseline_model_path(self) -> Path:
        return self.models_dir / "baseline_logistic_regression.joblib"

    @property
    def app_only_metrics_path(self) -> Path:
        return self.metrics_dir / "lightgbm_metrics.json"

    @property
    def app_only_model_path(self) -> Path:
        return self.models_dir / "lightgbm_challenger.joblib"

    @property
    def ablation_path(self) -> Path:
        return self.metrics_dir / "relational_ablation.csv"

    @property
    def selection_path(self) -> Path:
        return self.metrics_dir / "relational_selection.json"

    @property
    def trials_path(self) -> Path:
        return self.metrics_dir / "relational_trials.csv"

    @property
    def final_metrics_path(self) -> Path:
        return self.metrics_dir / f"{MODEL_KEY}_metrics.json"

    @property
    def comparison_path(self) -> Path:
        return self.metrics_dir / "relational_model_comparison.csv"

    @property
    def bootstrap_path(self) -> Path:
        return self.metrics_dir / "relational_bootstrap.json"

    @property
    def importance_path(self) -> Path:
        return self.metrics_dir / "relational_importance_by_source.csv"

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{MODEL_KEY}.joblib"

    def to_dict(self) -> dict[str, str]:
        return {k: _display_path(v) for k, v in asdict(self).items()}


@dataclass
class AblationConfig:
    valid_size: float = config.VALIDATION_SIZE
    random_state: int = config.RANDOM_STATE
    n_jobs: int = config.N_JOBS
    n_bootstrap_validation: int = 500
    bootstrap_seed: int = config.RANDOM_STATE
    n_calibration_bins: int = 10
    sources: tuple[str, ...] = SEQUENCE
    include_single: bool = True
    include_leave_one_out: bool = True
    param_overrides: dict[str, Any] = field(default_factory=dict)  # smoke runs only
    retune_axes: list[tuple[str, list[Any]]] = field(
        default_factory=lambda: [list(a) for a in RETUNE_AXES]
    )
    early_stopping_rounds: int = ch.EARLY_STOPPING_ROUNDS
    n_bootstrap_test: int = 1000

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["retune_axes"] = [[axis, list(values)] for axis, values in self.retune_axes]
        return out


def _run_info(seconds: float) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "riskpilot_version": __version__,
        "lightgbm_version": lgb.__version__,
        "total_seconds": round(seconds, 1),
    }


def locked_lightgbm_spec(
    selection_path: Path,
    *,
    n_jobs: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ch.ChallengerSpec:
    """The Milestone 2 locked LightGBM configuration (fixed rounds, no early stopping)."""
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"{selection_path} not found: run 'python -m riskpilot.models.challengers' first."
        )
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    spec = ch.ChallengerSpec.from_dict(payload["libraries"]["lightgbm"]["locked_spec"])
    params: dict[str, Any] = {}
    if n_jobs is not None:
        params["n_jobs"] = n_jobs
    if overrides:
        params.update(overrides)
    return spec.with_params(**params) if params else spec


# --------------------------------------------------------------------------- #
# Configurations and column selection
# --------------------------------------------------------------------------- #
def ablation_configs(
    sources: Sequence[str] = SEQUENCE,
    *,
    include_single: bool = True,
    include_leave_one_out: bool = True,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """``(name, group, sources)``: M0, the cumulative sequence, single-source, leave-one-out."""
    sources = tuple(sources)
    configs: list[tuple[str, str, tuple[str, ...]]] = [("M0_application", "sequential", ())]
    for i, source in enumerate(sources, start=1):
        configs.append((f"M{i}_+{source}", "sequential", sources[:i]))
    if include_single:
        for source in sources[1:]:  # S_<first> == M1
            configs.append((f"S_{source}", "single", (source,)))
    if include_leave_one_out and len(sources) > 1:
        seen = {subset for _, _, subset in configs}
        for source in sources:
            subset = tuple(s for s in sources if s != source)
            if subset in seen:  # leaving out the last source == the cumulative M(n-1) model
                continue
            configs.append((f"L-{source}", "leave-one-out", subset))
    return configs


def columns_for_sources(columns: Sequence[str], sources: Sequence[str]) -> list[str]:
    """Application columns plus the columns of the given sources, in frame order."""
    wanted = {"application", *sources}
    return [c for c in columns if assemble.feature_source(c) in wanted]


# --------------------------------------------------------------------------- #
# Fitting on the internal validation split
# --------------------------------------------------------------------------- #
def fit_and_score(
    spec: ch.ChallengerSpec,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    n_bins: int = 10,
) -> tuple[dict[str, Any], np.ndarray, ch.FitResult]:
    """Fit the frozen configuration (fixed rounds) and score the validation subset."""
    result = ch.fit_challenger(spec, X_fit, y_fit)  # no validation monitor: rounds are fixed
    p_valid = result.pipeline.predict_proba(X_valid)[:, 1]
    metrics = compute_metrics(y_valid, p_valid, n_bins=n_bins)
    record = {
        "n_features": result.n_features,
        "n_trees": result.n_trees,
        "fit_seconds": round(result.fit_seconds, 1),
        "matrix_mb": round(float(X_fit.memory_usage(deep=True).sum() / 1e6), 1),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["average_precision"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier_score"],
        "brier_skill": metrics["brier_skill_score"],
        "ece": metrics["expected_calibration_error"],
        "calibration_slope": metrics["calibration_slope"],
    }
    return record, p_valid, result


def backward_elimination(
    y_valid: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    records: Mapping[str, Mapping[str, Any]],
    sources: Sequence[str],
    *,
    evaluate_missing,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Drop, one at a time, the source whose removal costs nothing on validation.

    ``predictions`` / ``records`` hold validation predictions and metrics per
    configuration name; ``evaluate_missing(name, sources)`` fits a configuration
    that is not there yet and returns (record, p_valid). At each round the
    candidate is the source with the smallest removal cost in validation log
    loss; it is dropped when the paired-bootstrap interval of that cost
    includes zero, otherwise the elimination stops.
    """
    remaining = list(sources)
    full_name = _config_name(remaining, sources)
    rounds: list[dict[str, Any]] = []
    while len(remaining) > 1:
        candidates = []
        for source in remaining:
            subset = [s for s in remaining if s != source]
            name = _config_name(subset, sources)
            if name not in predictions:
                record, p_valid = evaluate_missing(name, tuple(subset))
                records[name] = record  # type: ignore[index]
                predictions[name] = p_valid  # type: ignore[index]
            boot = comparison.paired_bootstrap(
                y_valid,
                {"full": predictions[full_name], "without": predictions[name]},
                n_replicates=n_bootstrap,
                seed=seed,
                reference="full",
                metrics=("roc_auc", "log_loss"),
            )
            diff = boot["differences"]["without_vs_full"]
            candidates.append(
                {
                    "source": source,
                    "config": name,
                    "removal_cost_log_loss": diff["log_loss"]["estimate"],
                    "ci_low": diff["log_loss"]["ci_low"],
                    "ci_high": diff["log_loss"]["ci_high"],
                    "removal_cost_roc_auc": -diff["roc_auc"]["estimate"],
                    "roc_auc_ci_low": -diff["roc_auc"]["ci_high"],
                    "roc_auc_ci_high": -diff["roc_auc"]["ci_low"],
                    "fraction_worse_without": 1.0 - diff["log_loss"]["fraction_improving"],
                }
            )
        weakest = min(candidates, key=lambda c: c["removal_cost_log_loss"])
        drop = weakest["ci_low"] <= 0.0
        rounds.append(
            {
                "remaining_before": list(remaining),
                "full_config": full_name,
                "candidates": candidates,
                "weakest": weakest["source"],
                "dropped": drop,
            }
        )
        if not drop:
            break
        remaining.remove(weakest["source"])
        full_name = _config_name(remaining, sources)
    return {"retained": remaining, "final_config": full_name, "rounds": rounds}


def _config_name(subset: Sequence[str], sources: Sequence[str]) -> str:
    subset = list(subset)
    sources = list(sources)
    if not subset:
        return "M0_application"
    if subset == sources:
        return f"M{len(sources)}_+{sources[-1]}"
    if len(subset) == 1:
        return "M1_+" + subset[0] if subset[0] == sources[0] else f"S_{subset[0]}"
    if subset == sources[: len(subset)]:  # cumulative prefix (also covers leaving out the last)
        return f"M{len(subset)}_+{subset[-1]}"
    missing = [s for s in sources if s not in subset]
    if len(missing) == 1:
        return f"L-{missing[0]}"
    return "L-" + "-".join(missing)


# --------------------------------------------------------------------------- #
# Stage 1: source ablation on internal validation
# --------------------------------------------------------------------------- #
def _downcast(X: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 (trees cast anyway; halves the application block)."""
    out = X.copy(deep=False)
    for col in out.columns:
        if out[col].dtype == np.float64:
            out[col] = out[col].astype("float32")
    return out


def run_ablation_stage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ids_train: pd.Series,
    paths: RelationalPaths,
    cfg: AblationConfig | None = None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Source ablation on the internal validation split. Receives the training portion only."""
    cfg = cfg or AblationConfig()
    started = time.perf_counter()
    spec = locked_lightgbm_spec(
        paths.challenger_selection_path, n_jobs=cfg.n_jobs, overrides=cfg.param_overrides
    )
    logger.info(
        "Frozen LightGBM configuration: %s",
        {k: v for k, v in spec.params.items() if k != "verbose"},
    )

    X_fit, X_valid, y_fit, y_valid = make_split(
        X_train, y_train, test_size=cfg.valid_size, random_state=cfg.random_state
    )
    membership = pd.DataFrame({config.ID_COL: ids_train.to_numpy()}, index=X_train.index)
    membership["internal_split"] = "fit"
    membership.loc[X_valid.index, "internal_split"] = "validation"
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    membership.to_csv(paths.internal_membership_path, index=False)
    logger.info(
        "Internal split: fit=%d (prev %.4f) | validation=%d (prev %.4f) -> %s",
        len(y_fit),
        y_fit.mean(),
        len(y_valid),
        y_valid.mean(),
        _display_path(paths.internal_membership_path),
    )

    full = assemble.build_feature_matrix(
        _downcast(X_train), ids_train, cfg.sources, processed_dir=paths.relational_dir
    )
    coverage = {s: float(full[assemble.has_column(s)].mean()) for s in cfg.sources}
    logger.info(
        "Full matrix: %s x %d (%.0f MB) | history coverage %s",
        f"{len(full):,}",
        full.shape[1],
        full.memory_usage(deep=True).sum() / 1e6,
        {k: round(v, 3) for k, v in coverage.items()},
    )
    fit_index, valid_index = X_fit.index, X_valid.index
    del X_fit, X_valid

    records: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    if resume and paths.ablation_path.is_file() and paths.validation_predictions_path.is_file():
        old = pd.read_csv(paths.ablation_path)
        old_pred = pd.read_parquet(paths.validation_predictions_path)
        if len(old_pred) == len(valid_index):
            for row in old.to_dict(orient="records"):
                name = row["configuration"]
                if name in old_pred.columns:
                    records[name] = row
                    predictions[name] = old_pred[name].to_numpy()
            logger.info("Resumed %d ablation configurations", len(records))

    def evaluate(
        name: str, group: str, sources: tuple[str, ...]
    ) -> tuple[dict[str, Any], np.ndarray]:
        cols = columns_for_sources(full.columns, sources)
        t0 = time.perf_counter()
        record, p_valid, _ = fit_and_score(
            spec,
            full.loc[fit_index, cols],
            y_fit,
            full.loc[valid_index, cols],
            y_valid,
            n_bins=cfg.n_calibration_bins,
        )
        record = {
            "configuration": name,
            "group": group,
            "sources": "+".join(sources) if sources else "application",
            "n_sources": len(sources),
            **record,
            "wall_seconds": round(time.perf_counter() - t0, 1),
        }
        logger.info(
            "%-24s features %4d | val log loss %.5f | ROC-AUC %.4f | PR-AUC %.4f | %.0fs",
            name,
            record["n_features"],
            record["log_loss"],
            record["roc_auc"],
            record["pr_auc"],
            record["fit_seconds"],
        )
        return record, p_valid

    def persist() -> None:
        pd.DataFrame(list(records.values())).to_csv(paths.ablation_path, index=False)
        pd.DataFrame(predictions, index=valid_index).to_parquet(paths.validation_predictions_path)

    for name, group, sources in ablation_configs(
        cfg.sources,
        include_single=cfg.include_single,
        include_leave_one_out=cfg.include_leave_one_out,
    ):
        if name in records:
            continue
        records[name], predictions[name] = evaluate(name, group, sources)
        persist()

    y_valid_arr = y_valid.to_numpy()
    elimination = backward_elimination(
        y_valid_arr,
        predictions,
        records,
        cfg.sources,
        evaluate_missing=lambda name, sources: evaluate(name, "elimination", sources),
        n_bootstrap=cfg.n_bootstrap_validation,
        seed=cfg.bootstrap_seed,
    )
    persist()

    m0 = records["M0_application"]
    full_name = _config_name(cfg.sources, cfg.sources)
    source_value = {}
    for source in cfg.sources:
        single = records.get("M1_+" + source if source == cfg.sources[0] else f"S_{source}")
        loo = records.get(_config_name([s for s in cfg.sources if s != source], cfg.sources))
        source_value[source] = {
            "history_coverage": coverage[source],
            "n_features": int(
                len([c for c in full.columns if assemble.feature_source(c) == source])
            ),
            "single_source_gain_log_loss": (m0["log_loss"] - single["log_loss"])
            if single
            else None,
            "single_source_gain_roc_auc": (single["roc_auc"] - m0["roc_auc"]) if single else None,
            "leave_one_out_cost_log_loss": (loo["log_loss"] - records[full_name]["log_loss"])
            if loo
            else None,
            "leave_one_out_cost_roc_auc": (records[full_name]["roc_auc"] - loo["roc_auc"])
            if loo
            else None,
        }
    retained = elimination["retained"]
    final_name = elimination["final_config"]
    selection = {
        "run": _run_info(time.perf_counter() - started),
        "config": cfg.to_dict(),
        "frozen_lightgbm_spec": spec.to_dict(),
        "validation_split": {
            "strategy": "stratified holdout inside the frozen training portion (as Milestone 2)",
            "membership_file": _display_path(paths.internal_membership_path),
            "n_fit": int(len(fit_index)),
            "n_valid": int(len(valid_index)),
            "prevalence_fit": float(y_fit.mean()),
            "prevalence_valid": float(y_valid.mean()),
        },
        "full_matrix": {
            "n_rows": int(len(full)),
            "n_features": int(full.shape[1]),
            "history_coverage": coverage,
        },
        "source_value": source_value,
        "elimination": elimination,
        "retained_sources": retained,
        "retained_config": final_name,
        "retained_validation_metrics": {
            k: records[final_name][k]
            for k in ("n_features", "roc_auc", "pr_auc", "log_loss", "brier", "brier_skill", "ece")
        },
        "application_only_validation_metrics": {
            k: m0[k]
            for k in ("n_features", "roc_auc", "pr_auc", "log_loss", "brier", "brier_skill", "ece")
        },
        "artifacts": {
            "ablation": _display_path(paths.ablation_path),
            "validation_predictions": _display_path(paths.validation_predictions_path),
        },
    }
    save_metrics(selection, paths.selection_path)
    logger.info("Retained sources: %s (%s)", retained, final_name)
    return selection


# --------------------------------------------------------------------------- #
# Stage 2: light retuning on the retained feature set
# --------------------------------------------------------------------------- #
def run_retune_stage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ids_train: pd.Series,
    selection: dict[str, Any],
    paths: RelationalPaths,
    cfg: AblationConfig | None = None,
) -> dict[str, Any]:
    """Small coordinate search with early stopping on validation, retained sources only."""
    cfg = cfg or AblationConfig()
    started = time.perf_counter()
    base = locked_lightgbm_spec(
        paths.challenger_selection_path, n_jobs=cfg.n_jobs, overrides=cfg.param_overrides
    ).with_params(n_estimators=cfg.param_overrides.get("n_estimators", ch.MAX_BOOSTING_ROUNDS))
    retained = tuple(selection["retained_sources"])
    X_fit, X_valid, y_fit, y_valid = make_split(
        X_train, y_train, test_size=cfg.valid_size, random_state=cfg.random_state
    )
    full = assemble.build_feature_matrix(
        _downcast(X_train), ids_train, retained, processed_dir=paths.relational_dir
    )
    cols = columns_for_sources(full.columns, retained)
    X_fit_r, X_valid_r = full.loc[X_fit.index, cols], full.loc[X_valid.index, cols]
    del full, X_fit, X_valid

    log = ch.TrialLog(paths.trials_path)
    best_spec, best = ch.coordinate_search(
        base,
        cfg.retune_axes,
        X_fit_r,
        y_fit,
        X_valid_r,
        y_valid,
        log=log,
        early_stopping_rounds=cfg.early_stopping_rounds,
        n_bins=cfg.n_calibration_bins,
    )
    locked = best_spec.with_params(n_estimators=int(best["n_trees"]))
    frozen = selection["frozen_lightgbm_spec"]["params"]
    changed = {
        k: {"frozen": frozen.get(k), "retuned": locked.params.get(k)}
        for k in (
            "num_leaves",
            "min_child_samples",
            "colsample_bytree",
            "reg_lambda",
            "learning_rate",
            "n_estimators",
        )
        if frozen.get(k) != locked.params.get(k)
    }
    frozen_record = records_for_frozen(selection)
    retune = {
        "run": _run_info(time.perf_counter() - started),
        "axes": [[axis, list(values)] for axis, values in cfg.retune_axes],
        "n_trials": len(log.records),
        "selected_trial": best["trial_id"],
        "locked_spec": locked.to_dict(),
        "changed_vs_frozen": changed,
        "validation_metrics": {k: best[f"val_{k}"] for k in ch.VAL_METRIC_KEYS},
        "frozen_configuration_validation_metrics": frozen_record,
        "artifacts": {"trials": _display_path(paths.trials_path)},
    }
    selection["retune"] = retune
    save_metrics(selection, paths.selection_path)
    logger.info(
        "Retuning: %d trials, best %s (val log loss %.5f vs frozen %.5f), changes %s",
        len(log.records),
        best["trial_id"],
        best["val_log_loss"],
        frozen_record["log_loss"],
        list(changed),
    )
    return retune


def records_for_frozen(selection: Mapping[str, Any]) -> dict[str, Any]:
    return dict(selection["retained_validation_metrics"])


# --------------------------------------------------------------------------- #
# Stage 3: locked feature set + configuration on the frozen test split
# --------------------------------------------------------------------------- #
def importance_by_source(model: lgb.LGBMClassifier, feature_names: Sequence[str]) -> pd.DataFrame:
    """Tree-based (gain and split) importance aggregated by feature source. Not a causal measure."""
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    frame = pd.DataFrame({"feature": list(feature_names), "gain": gain, "split": split})
    frame["source"] = [assemble.feature_source(f) for f in frame["feature"]]
    out = frame.groupby("source").agg(
        n_features=("feature", "size"), gain=("gain", "sum"), split=("split", "sum")
    )
    out["gain_share"] = out["gain"] / out["gain"].sum()
    out["split_share"] = out["split"] / out["split"].sum()
    out["features_used"] = (
        frame[frame["split"] > 0]
        .groupby("source")["feature"]
        .size()
        .reindex(out.index)
        .fillna(0)
        .astype(int)
    )
    top = frame.sort_values("gain", ascending=False).groupby("source")["feature"].head(3)
    out["top_features_by_gain"] = (
        frame.loc[top.index].groupby("source")["feature"].apply(lambda s: ", ".join(s))
    )
    order = ["application", *SOURCES]
    return out.reindex([s for s in order if s in out.index]).reset_index()


def run_final_stage(
    split: comparison.FrozenSplit,
    selection: Mapping[str, Any],
    paths: RelationalPaths,
    cfg: AblationConfig | None = None,
) -> dict[str, Any]:
    """Refit the locked configuration on the whole training portion; score the test split once."""
    cfg = cfg or AblationConfig()
    started = time.perf_counter()
    retained = tuple(selection["retained_sources"])
    spec_payload = (
        selection.get("retune", {}).get("locked_spec") or selection["frozen_lightgbm_spec"]
    )
    spec = ch.ChallengerSpec.from_dict(spec_payload).with_params(
        n_jobs=cfg.n_jobs, **cfg.param_overrides
    )
    if "n_estimators" in cfg.param_overrides:
        spec = spec.with_params(n_estimators=cfg.param_overrides["n_estimators"])

    X_train = assemble.build_feature_matrix(
        _downcast(split.X_train), split.ids_train, retained, processed_dir=paths.relational_dir
    )
    X_test = assemble.build_feature_matrix(
        _downcast(split.X_test), split.ids_test, retained, processed_dir=paths.relational_dir
    )
    cols = columns_for_sources(X_train.columns, retained)
    X_train, X_test = X_train[cols], X_test[cols]
    logger.info(
        "Final relational model: %d features, sources %s, %d rounds",
        len(cols),
        retained,
        spec.params["n_estimators"],
    )

    baseline = comparison.load_baseline_reference(
        split, model_path=paths.baseline_model_path, metrics_path=paths.baseline_metrics_path
    )
    app_only = comparison.load_model_reference(
        split,
        model_path=paths.app_only_model_path,
        metrics_path=paths.app_only_metrics_path,
        label=APP_ONLY_KEY,
    )

    result = ch.fit_challenger(spec, X_train, split.y_train)
    t0 = time.perf_counter()
    p_test = result.pipeline.predict_proba(X_test)[:, 1]
    predict_seconds = time.perf_counter() - t0
    p_train = result.pipeline.predict_proba(X_train)[:, 1]
    m_test = compute_metrics(split.y_test, p_test, n_bins=cfg.n_calibration_bins)
    m_train = compute_metrics(split.y_train, p_train, n_bins=cfg.n_calibration_bins)
    table = calibration_table(split.y_test, p_test, n_bins=cfg.n_calibration_bins)
    figures = make_evaluation_figures(
        split.y_test,
        p_test,
        figures_dir=paths.figures_dir,
        prefix=MODEL_KEY,
        n_bins=cfg.n_calibration_bins,
    )
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, paths.model_path)
    model_size_mb = round(paths.model_path.stat().st_size / 1e6, 3)
    importance = importance_by_source(result.model, result.preprocessor.get_feature_names_out())
    importance.to_csv(paths.importance_path, index=False)

    summaries = {
        "logistic_regression": baseline.summary,
        APP_ONLY_KEY: app_only.summary,
        MODEL_KEY: {
            "label": comparison.LABELS[MODEL_KEY],
            "n_features": result.n_features,
            "n_trees": result.n_trees,
            "fit_seconds": round(result.fit_seconds, 2),
            "predict_seconds_test": round(predict_seconds, 3),
            "model_size_mb": model_size_mb,
            "metrics_test": m_test,
            "metrics_train": m_train,
        },
    }
    predictions = {
        "logistic_regression": baseline.y_prob_test,
        APP_ONLY_KEY: app_only.y_prob_test,
        MODEL_KEY: p_test,
    }
    table_df = comparison.comparison_table(summaries, reference=APP_ONLY_KEY)
    lr = baseline.summary["metrics_test"]
    for k in comparison.BOOTSTRAP_METRICS:
        table_df[f"delta_vs_lr_{k}"] = [s["metrics_test"][k] - lr[k] for s in summaries.values()]
    table_df.to_csv(paths.comparison_path, index=False)

    boot = comparison.paired_bootstrap(
        split.y_test,
        predictions,
        n_replicates=cfg.n_bootstrap_test,
        seed=cfg.bootstrap_seed,
        reference=APP_ONLY_KEY,
    )
    boot["run"] = _run_info(time.perf_counter() - started)
    save_metrics(boot, paths.bootstrap_path)
    comp_figures = comparison.make_comparison_figures(
        split.y_test,
        predictions,
        boot,
        figures_dir=paths.figures_dir,
        prefix="relational",
        n_bins=cfg.n_calibration_bins,
    )
    ablation_figure = paths.figures_dir / "relational_ablation_validation.png"
    importance_figure = paths.figures_dir / "relational_importance_by_source.png"
    plot_ablation(pd.read_csv(paths.ablation_path), path=ablation_figure)
    plot_importance_by_source(importance, path=importance_figure)
    comp_figures["ablation_validation"] = ablation_figure
    comp_figures["importance_by_source"] = importance_figure
    plt.close("all")

    out = pd.DataFrame(
        {split.ids_test.name or "row": split.ids_test.to_numpy(), "y": split.y_test.to_numpy()}
    )
    for name, p in predictions.items():
        out[f"p_{name}"] = p
    out.to_csv(paths.test_predictions_path, index=False)

    payload: dict[str, Any] = {
        "run": _run_info(time.perf_counter() - started),
        "model_key": MODEL_KEY,
        "label": comparison.LABELS[MODEL_KEY],
        "spec": spec.to_dict(),
        "retained_sources": list(retained),
        "selection_file": _display_path(paths.selection_path),
        "data": {
            "n_rows": int(len(split.y_train) + len(split.y_test)),
            "n_application_features": int(split.X_train.shape[1]),
            "n_relational_features": int(len(cols) - split.X_train.shape[1]),
            "n_features": int(len(cols)),
            "preprocessing": result.preprocessor.describe(),
        },
        "split": split.summary(),
        "references": {
            "logistic_regression": baseline.integrity,
            APP_ONLY_KEY: app_only.integrity,
        },
        "model": {
            "estimator": "LGBMClassifier",
            "library_version": lgb.__version__,
            "n_trees": result.n_trees,
            "early_stopping": "none (rounds locked during selection)",
            "fit_seconds": round(result.fit_seconds, 2),
            "predict_seconds_test": round(predict_seconds, 3),
            "n_features": result.n_features,
            "model_size_mb": model_size_mb,
            "class_weight": None,
            "n_jobs": spec.params.get("n_jobs"),
        },
        "metrics": {"test": m_test, "train": m_train},
        "calibration_table_test": table.to_dict(orient="records"),
        "importance_by_source": importance.to_dict(orient="records"),
        "artifacts": {
            "figures": {k: _display_path(v) for k, v in {**figures, **comp_figures}.items()},
            "model": _display_path(paths.model_path),
            "comparison_table": _display_path(paths.comparison_path),
            "bootstrap": _display_path(paths.bootstrap_path),
            "importance": _display_path(paths.importance_path),
            "test_predictions": _display_path(paths.test_predictions_path),
        },
    }
    save_metrics(payload, paths.final_metrics_path)
    logger.info(
        "Final %s on the frozen test split: ROC-AUC %.4f | PR-AUC %.4f | log loss %.5f | "
        "Brier %.5f | ECE %.4f",
        MODEL_KEY,
        m_test["roc_auc"],
        m_test["average_precision"],
        m_test["log_loss"],
        m_test["brier_score"],
        m_test["expected_calibration_error"],
    )
    return {
        "payload": payload,
        "comparison_table": table_df,
        "bootstrap": boot,
        "predictions": predictions,
        "importance": importance,
        "references": {"logistic_regression": baseline, APP_ONLY_KEY: app_only},
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_ablation(ablation: pd.DataFrame, *, path: str | Path | None = None) -> Figure:
    """Validation ROC-AUC and log loss per configuration, coloured by configuration group."""
    order = ["sequential", "single", "leave-one-out", "elimination"]
    frame = ablation.copy()
    frame["group_rank"] = frame["group"].map({g: i for i, g in enumerate(order)}).fillna(len(order))
    frame = frame.sort_values(["group_rank", "configuration"], kind="stable").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(frame) + 1.8), sharey=True)
    y = np.arange(len(frame))
    for ax, metric, label in zip(
        axes,
        ("roc_auc", "log_loss"),
        ("validation ROC-AUC (higher is better)", "validation log loss (lower is better)"),
        strict=True,
    ):
        ref = float(frame.loc[frame["configuration"] == "M0_application", metric].iloc[0])
        ax.axvline(
            ref, color=REFERENCE_COLOR, linewidth=1, linestyle="--", label="application only (M0)"
        )
        for group in order:
            mask = frame["group"] == group
            if not mask.any():
                continue
            ax.scatter(
                frame.loc[mask, metric],
                y[mask.to_numpy()],
                s=64,
                color=GROUP_COLORS.get(group, REFERENCE_COLOR),
                edgecolor="white",
                linewidth=1.5,
                label=group,
                zorder=3,
            )
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        _style_axes(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [
            f"{c}  ({n} feat.)"
            for c, n in zip(frame["configuration"], frame["n_features"], strict=True)
        ],
        fontsize=8.5,
    )
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Source ablation on internal validation, frozen LightGBM configuration", y=1.0)
    return _finish(fig, path)


def plot_importance_by_source(
    importance: pd.DataFrame, *, path: str | Path | None = None
) -> Figure:
    """Gain share by source (tree-based importance; an engineering sanity check, not causal)."""
    frame = importance.sort_values("gain_share", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(frame) + 1.6))
    ax.barh(frame["source"], frame["gain_share"], color=MODEL_COLOR, height=0.6)
    for i, (share, n, used) in enumerate(
        zip(frame["gain_share"], frame["n_features"], frame["features_used"], strict=True)
    ):
        ax.text(
            share + 0.005,
            i,
            f"{share:.1%}  ({used}/{n} features used)",
            va="center",
            fontsize=9,
            color="#52514e",
        )
    ax.set_xlim(0, max(0.05, float(frame["gain_share"].max()) * 1.45))
    ax.set_xlabel("share of total LightGBM gain (final relational model)")
    ax.set_title("Tree-based importance by feature source")
    ax.grid(axis="y", visible=False)
    _style_axes(ax)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_split(paths: RelationalPaths, *, nrows: int | None = None) -> comparison.FrozenSplit:
    exp_paths = ch.ExperimentPaths(data_path=paths.data_path, membership_path=paths.membership_path)
    return ch.load_frozen_split(exp_paths, nrows=nrows)


def run_experiment(
    paths: RelationalPaths,
    *,
    stage: str = "all",
    cfg: AblationConfig | None = None,
    nrows: int | None = None,
    resume: bool = False,
    retune: bool = True,
) -> dict[str, Any]:
    cfg = cfg or AblationConfig()
    split = load_split(paths, nrows=nrows)
    out: dict[str, Any] = {"split": split.summary()}
    selection: dict[str, Any] | None = None
    if stage in ("all", "ablation"):
        selection = run_ablation_stage(
            split.X_train, split.y_train, split.ids_train, paths, cfg, resume=resume
        )
        out["selection"] = selection
    if stage in ("all", "retune") and retune:
        selection = selection or json.loads(paths.selection_path.read_text(encoding="utf-8"))
        out["retune"] = run_retune_stage(
            split.X_train, split.y_train, split.ids_train, selection, paths, cfg
        )
    if stage in ("all", "final"):
        selection = selection or json.loads(paths.selection_path.read_text(encoding="utf-8"))
        out["final"] = run_final_stage(split, selection, paths, cfg)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riskpilot.models.relational_experiment",
        description="Relational feature ablation, light retuning and frozen-test evaluation.",
    )
    parser.add_argument("--stage", choices=["all", "ablation", "retune", "final"], default="all")
    parser.add_argument("--no-retune", action="store_true", help="Skip the light retuning stage.")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse ablation configurations already recorded."
    )
    parser.add_argument("--quick", action="store_true", help="Tiny models and search (smoke runs).")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--data-path", type=Path, default=config.APPLICATION_TRAIN_PATH)
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DATA_DIR)
    parser.add_argument("--relational-dir", type=Path, default=config.RELATIONAL_PROCESSED_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=config.METRICS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=config.FIGURES_DIR)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    use_headless_backend()
    args = _build_parser().parse_args(argv)
    paths = RelationalPaths(
        data_path=args.data_path,
        processed_dir=args.processed_dir,
        relational_dir=args.relational_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        models_dir=args.models_dir,
    )
    cfg = AblationConfig(n_jobs=args.n_jobs, n_bootstrap_test=args.n_bootstrap)
    if args.quick:
        cfg = AblationConfig(
            n_jobs=args.n_jobs,
            n_bootstrap_test=args.n_bootstrap,
            n_bootstrap_validation=30,
            param_overrides={"n_estimators": 40, "min_child_samples": 5, "num_leaves": 7},
            retune_axes=[list(a) for a in SMOKE_RETUNE_AXES],
            early_stopping_rounds=10,
        )
    try:
        out = run_experiment(
            paths,
            stage=args.stage,
            cfg=cfg,
            nrows=args.nrows,
            resume=args.resume,
            retune=not args.no_retune,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    if "final" in out:
        table = out["final"]["comparison_table"]
        cols = [
            "model",
            "n_features",
            "roc_auc",
            "average_precision",
            "log_loss",
            "brier_score",
            "brier_skill_score",
            "expected_calibration_error",
        ]
        print(
            "\nFrozen test split comparison\n"
            + table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )
        print(
            "\nReferences: "
            + json.dumps(
                {
                    k: v["metrics_match_recorded"]
                    for k, v in out["final"]["payload"]["references"].items()
                }
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
