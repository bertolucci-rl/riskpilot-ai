"""Gradient-boosting challengers (LightGBM, XGBoost) against the frozen baseline.

Run from the command line::

    python -m riskpilot.models.challengers                 # selection + final stage
    python -m riskpilot.models.challengers --stage select  # internal validation only
    python -m riskpilot.models.challengers --stage final   # locked configs -> frozen test

Experimental protocol (the report explains the reasoning):

* The Milestone 1 split is reloaded from ``data/processed/baseline_split_membership.csv``
  and verified against a re-derivation from the seed. The frozen **test** partition
  is touched only by :func:`run_final_stage`, once per locked configuration.
* :func:`run_selection_stage` receives the frozen **training** portion only. It
  carves a stratified validation subset out of it (``config.VALIDATION_SIZE``),
  fits the Logistic Regression reference on the remainder, runs a compact
  coordinate-descent search per library with early stopping on the validation
  log loss, then two controlled preprocessing ablations, and locks one
  configuration per library (including the number of boosting rounds).
* Every fitted configuration is appended to a machine-readable trial log.
* :func:`run_final_stage` refits each locked configuration on the whole training
  portion without early stopping, scores the frozen test partition once, and
  hands the predictions to :mod:`riskpilot.models.comparison`.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.pipeline import Pipeline

from riskpilot import __version__, config
from riskpilot.data.load import categorize_strings, load_application_train, split_features_target
from riskpilot.data.validation import validate_target
from riskpilot.features.preprocessing import build_baseline_pipeline, infer_feature_types
from riskpilot.features.tree_preprocessing import HeavyTail, MissingIndicators, TreePreprocessor
from riskpilot.models import comparison
from riskpilot.models.evaluate import (
    calibration_table,
    compute_metrics,
    make_evaluation_figures,
    save_metrics,
    use_headless_backend,
)
from riskpilot.models.train import _display_path, make_split

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PARAMS",
    "LIBRARIES",
    "SEARCH_AXES",
    "SMOKE_SEARCH_AXES",
    "ChallengerSpec",
    "ExperimentPaths",
    "FinalConfig",
    "FitResult",
    "SelectionConfig",
    "TrialLog",
    "build_challenger",
    "build_estimator",
    "build_preprocessor",
    "coordinate_search",
    "evaluate_spec",
    "fit_challenger",
    "lock_spec",
    "main",
    "run_final_stage",
    "run_selection_stage",
]

LIBRARIES: tuple[str, ...] = ("lightgbm", "xgboost")
LIBRARY_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}
MAX_BOOSTING_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 100
SELECTION_METRIC = "log_loss"  # proper scoring rule: rewards ranking *and* probability quality

# Sensible, minimally tuned starting points (no class re-weighting, by design).
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "objective": "binary",
        "learning_rate": 0.05,
        "n_estimators": MAX_BOOSTING_ROUNDS,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": config.RANDOM_STATE,
        "n_jobs": config.N_JOBS,
        "verbose": 0,  # warnings stay visible; only info logging is off
    },
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "n_estimators": MAX_BOOSTING_ROUNDS,
        "max_depth": 6,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "enable_categorical": True,
        "random_state": config.RANDOM_STATE,
        "n_jobs": config.N_JOBS,
        "verbosity": 1,  # warnings stay visible
    },
}

# Coordinate-descent axes, in order. Each axis is swept holding the others at
# their current best; the winner (validation log loss) is carried forward.
SEARCH_AXES: dict[str, list[tuple[str, list[Any]]]] = {
    "lightgbm": [
        ("num_leaves", [15, 31, 63]),
        ("min_child_samples", [20, 100, 500]),
        ("colsample_bytree", [0.3, 0.5, 0.8]),
        ("subsample", [0.6, 0.8, 1.0]),
        ("reg_lambda", [0.0, 1.0, 10.0]),
        ("learning_rate", [0.02, 0.05]),
    ],
    "xgboost": [
        ("max_depth", [3, 4, 6]),
        ("min_child_weight", [1, 10, 100]),
        ("colsample_bytree", [0.3, 0.5, 0.8]),
        ("subsample", [0.6, 0.8, 1.0]),
        ("reg_lambda", [1.0, 10.0]),
        ("reg_alpha", [0.0, 1.0]),
        ("learning_rate", [0.02, 0.05]),
    ],
}
# Tiny grid for smoke runs and unit tests.
SMOKE_SEARCH_AXES: dict[str, list[tuple[str, list[Any]]]] = {
    "lightgbm": [("num_leaves", [7, 15])],
    "xgboost": [("max_depth", [2, 3])],
}
SMOKE_PARAM_OVERRIDES: dict[str, dict[str, Any]] = {
    "lightgbm": {"n_estimators": 60, "min_child_samples": 5, "n_jobs": 2},
    "xgboost": {"n_estimators": 60, "min_child_weight": 1, "n_jobs": 2},
}

VAL_METRIC_KEYS: tuple[str, ...] = (
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "brier_skill_score",
    "expected_calibration_error",
    "calibration_slope",
    "calibration_intercept",
    "mean_predicted_probability",
)
TRAIN_METRIC_KEYS: tuple[str, ...] = ("roc_auc", "average_precision", "log_loss", "brier_score")


# --------------------------------------------------------------------------- #
# Specification, construction, fitting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChallengerSpec:
    """One fully determined challenger: library, estimator params, preprocessing."""

    library: str
    params: dict[str, Any]
    heavy_tail: HeavyTail = "none"
    missing_indicators: MissingIndicators = "none"

    def __post_init__(self) -> None:
        if self.library not in LIBRARIES:
            raise ValueError(f"library must be one of {LIBRARIES}, got {self.library!r}.")

    @classmethod
    def default(cls, library: str, **overrides: Any) -> ChallengerSpec:
        return cls(library=library, params={**DEFAULT_PARAMS[library], **overrides})

    def with_params(self, **overrides: Any) -> ChallengerSpec:
        return replace(self, params={**self.params, **overrides})

    def with_preprocessing(
        self,
        *,
        heavy_tail: HeavyTail | None = None,
        missing_indicators: MissingIndicators | None = None,
    ) -> ChallengerSpec:
        return replace(
            self,
            heavy_tail=self.heavy_tail if heavy_tail is None else heavy_tail,
            missing_indicators=(
                self.missing_indicators if missing_indicators is None else missing_indicators
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "library": self.library,
            "params": dict(self.params),
            "heavy_tail": self.heavy_tail,
            "missing_indicators": self.missing_indicators,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChallengerSpec:
        return cls(
            library=payload["library"],
            params=dict(payload["params"]),
            heavy_tail=payload.get("heavy_tail", "none"),
            missing_indicators=payload.get("missing_indicators", "none"),
        )


def build_preprocessor(spec: ChallengerSpec) -> TreePreprocessor:
    return TreePreprocessor(
        heavy_tail=spec.heavy_tail,
        missing_indicators=spec.missing_indicators,
    )


def build_estimator(
    spec: ChallengerSpec,
    *,
    early_stopping_rounds: int | None = None,
) -> lgb.LGBMClassifier | xgb.XGBClassifier:
    """Instantiate the estimator. Early stopping is configured only when requested."""
    params = dict(spec.params)
    if spec.library == "lightgbm":
        return lgb.LGBMClassifier(**params)
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
    return xgb.XGBClassifier(**params)


def build_challenger(spec: ChallengerSpec, *, early_stopping_rounds: int | None = None) -> Pipeline:
    """``raw features -> TreePreprocessor -> booster`` (unfitted)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(spec)),
            ("model", build_estimator(spec, early_stopping_rounds=early_stopping_rounds)),
        ]
    )


@dataclass
class FitResult:
    pipeline: Pipeline
    spec: ChallengerSpec
    n_trees: int
    best_iteration: int | None
    early_stopped: bool
    fit_seconds: float
    n_features: int

    @property
    def model(self) -> lgb.LGBMClassifier | xgb.XGBClassifier:
        return self.pipeline["model"]

    @property
    def preprocessor(self) -> TreePreprocessor:
        return self.pipeline["preprocess"]


def _n_trees(model: lgb.LGBMClassifier | xgb.XGBClassifier) -> int:
    if isinstance(model, lgb.LGBMClassifier):
        return int(model.booster_.num_trees())
    return int(model.get_booster().num_boosted_rounds())


def fit_challenger(
    spec: ChallengerSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> FitResult:
    """Fit preprocessing + booster on ``X_train``; optionally early-stop on ``X_valid``.

    The preprocessor is fitted on ``X_train`` only and then applied to the
    validation frame, so nothing (vocabularies, clipping bounds, indicator
    groups) is learned from validation rows. The validation frame is used
    exclusively as the early-stopping monitor (log loss).
    """
    use_validation = X_valid is not None
    if use_validation and y_valid is None:
        raise ValueError("y_valid is required when X_valid is given.")

    pipeline = build_challenger(
        spec, early_stopping_rounds=early_stopping_rounds if use_validation else None
    )
    pre: TreePreprocessor = pipeline["preprocess"]
    model = pipeline["model"]

    started = time.perf_counter()
    Xt = pre.fit_transform(X_train)
    best_iteration: int | None = None
    early_stopped = False
    if use_validation:
        Xv = pre.transform(X_valid)
        if spec.library == "lightgbm":
            model.fit(
                Xt,
                y_train,
                eval_X=Xv,  # lightgbm >= 4.7: a single frame (a list is parsed as raw data)
                eval_y=y_valid,
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            best_iteration = int(model.best_iteration_)
            early_stopped = _n_trees(model) < spec.params["n_estimators"] or (
                best_iteration < _n_trees(model)
            )
            n_trees = best_iteration  # predict_proba defaults to best_iteration_
        else:
            model.fit(Xt, y_train, eval_set=[(Xv, y_valid)], verbose=False)
            best_iteration = int(model.best_iteration) + 1  # xgboost reports a 0-based index
            early_stopped = _n_trees(model) < spec.params["n_estimators"]
            n_trees = best_iteration  # predict_proba uses (0, best_iteration + 1)
        if not early_stopped:
            warnings.warn(
                f"{spec.library}: early stopping never triggered within "
                f"{spec.params['n_estimators']} rounds; the learning rate may be too low.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        if spec.library == "lightgbm":
            model.fit(Xt, y_train)
        else:
            model.fit(Xt, y_train, verbose=False)
        n_trees = _n_trees(model)
    fit_seconds = time.perf_counter() - started

    return FitResult(
        pipeline=pipeline,
        spec=spec,
        n_trees=int(n_trees),
        best_iteration=best_iteration,
        early_stopped=early_stopped,
        fit_seconds=fit_seconds,
        n_features=int(pre.get_feature_names_out().shape[0]),
    )


def lock_spec(result: FitResult) -> ChallengerSpec:
    """Freeze the number of boosting rounds found by early stopping.

    The locked spec is refitted on the whole training portion *without* early
    stopping, so the frozen test partition never acts as a monitor.
    """
    return result.spec.with_params(n_estimators=int(result.n_trees))


# --------------------------------------------------------------------------- #
# Trial log
# --------------------------------------------------------------------------- #
class TrialLog:
    """Machine-readable record of every configuration evaluated on validation data.

    With ``resume=True`` an existing CSV is loaded first and
    :func:`evaluate_spec` reuses the recorded validation metrics of any
    configuration it has already scored instead of refitting it. The search is
    deterministic, so an interrupted selection stage continues from where it
    stopped without changing its outcome; the CSV never contains duplicates.
    """

    def __init__(self, path: Path | None = None, *, resume: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        self.records: list[dict[str, Any]] = []
        self.reused: list[str] = []
        if resume and self.path is not None and self.path.is_file():
            frame = pd.read_csv(self.path)
            self.records = [
                {k: (None if pd.isna(v) else v) for k, v in row.items()}
                for row in frame.to_dict(orient="records")
            ]
            logger.info("Resuming from %s: %d recorded trials", self.path, len(self.records))

    @staticmethod
    def spec_key(spec: ChallengerSpec) -> tuple[str, str, str, str]:
        return (
            spec.library,
            json.dumps(spec.params, sort_keys=True, default=str),
            spec.heavy_tail,
            spec.missing_indicators,
        )

    def find(self, spec: ChallengerSpec) -> dict[str, Any] | None:
        """The recorded trial of an identical configuration, if any."""
        key = self.spec_key(spec)
        for record in self.records:
            if (
                record.get("library"),
                record.get("params_json"),
                record.get("heavy_tail"),
                record.get("missing_indicators"),
            ) == key:
                return record
        return None

    def find_id(self, trial_id: str) -> dict[str, Any] | None:
        return next((r for r in self.records if r.get("trial_id") == trial_id), None)

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.frame().to_csv(self.path, index=False)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def next_id(self, library: str) -> str:
        n = sum(1 for r in self.records if r["library"] == library)
        return f"{library}-{n + 1:02d}"


def _flatten_metrics(
    prefix: str, metrics: Mapping[str, Any], keys: Sequence[str]
) -> dict[str, Any]:
    return {f"{prefix}_{k}": metrics.get(k) for k in keys}


def evaluate_spec(
    spec: ChallengerSpec,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    log: TrialLog,
    stage: str,
    description: str = "",
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Fit one spec with early stopping and record its validation metrics.

    If ``log`` already holds a record of this exact configuration (a resumed
    run), that record is returned and nothing is refitted.
    """
    cached = log.find(spec)
    if cached is not None:
        log.reused.append(str(cached["trial_id"]))
        logger.info(
            "%s [%s] %s: reusing recorded trial", cached["trial_id"], stage, description or "-"
        )
        return cached
    result = fit_challenger(
        spec,
        X_fit,
        y_fit,
        X_valid=X_valid,
        y_valid=y_valid,
        early_stopping_rounds=early_stopping_rounds,
    )
    p_valid = result.pipeline.predict_proba(X_valid)[:, 1]
    p_fit = result.pipeline.predict_proba(X_fit)[:, 1]
    m_valid = compute_metrics(y_valid, p_valid, n_bins=n_bins)
    m_fit = compute_metrics(y_fit, p_fit, n_bins=n_bins)
    record: dict[str, Any] = {
        "trial_id": log.next_id(spec.library),
        "library": spec.library,
        "stage": stage,
        "description": description,
        "heavy_tail": spec.heavy_tail,
        "missing_indicators": spec.missing_indicators,
        "n_features": result.n_features,
        "n_trees": result.n_trees,
        "early_stopped": result.early_stopped,
        "fit_seconds": round(result.fit_seconds, 2),
        **_flatten_metrics("val", m_valid, VAL_METRIC_KEYS),
        **_flatten_metrics("train", m_fit, TRAIN_METRIC_KEYS),
        "params_json": json.dumps(spec.params, sort_keys=True, default=str),
    }
    log.append(record)
    logger.info(
        "%s [%s] %s: val log loss %.5f | ROC-AUC %.4f | AP %.4f | trees %d | %.1fs",
        record["trial_id"],
        stage,
        description or "-",
        m_valid["log_loss"],
        m_valid["roc_auc"],
        m_valid["average_precision"],
        result.n_trees,
        result.fit_seconds,
    )
    return record


# --------------------------------------------------------------------------- #
# Search and ablations (validation data only)
# --------------------------------------------------------------------------- #
def _better(record: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    return record[f"val_{SELECTION_METRIC}"] < incumbent[f"val_{SELECTION_METRIC}"]


def coordinate_search(
    base: ChallengerSpec,
    axes: Sequence[tuple[str, Sequence[Any]]],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    log: TrialLog,
    **eval_kwargs: Any,
) -> tuple[ChallengerSpec, dict[str, Any]]:
    """Compact coordinate descent: sweep one axis at a time, keep the best.

    Returns the best spec and its trial record. The value the axis starts from is
    already scored (by the incumbent record) and is not refitted.
    """
    incumbent_spec = base
    incumbent = evaluate_spec(
        base,
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        log=log,
        stage="initial",
        description="starting configuration",
        **eval_kwargs,
    )
    for axis, values in axes:
        start_value = incumbent_spec.params.get(axis)  # already scored by the incumbent record
        for value in values:
            if value == start_value:
                continue
            candidate = incumbent_spec.with_params(**{axis: value})
            record = evaluate_spec(
                candidate,
                X_fit,
                y_fit,
                X_valid,
                y_valid,
                log=log,
                stage=f"search:{axis}",
                description=f"{axis}={value}",
                **eval_kwargs,
            )
            if _better(record, incumbent):
                incumbent, incumbent_spec = record, candidate
    return incumbent_spec, incumbent


def _ablation(
    name: str,
    incumbent_spec: ChallengerSpec,
    incumbent: dict[str, Any],
    variants: Sequence[ChallengerSpec],
    labels: Sequence[str],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    log: TrialLog,
    min_gain: float,
    **eval_kwargs: Any,
) -> tuple[ChallengerSpec, dict[str, Any], dict[str, Any]]:
    """Controlled comparison: incumbent vs. variants; adopt only a clear gain.

    A variant is adopted when it lowers the validation log loss by at least
    ``min_gain`` (parsimony: added preprocessing must earn its place).
    """
    records: dict[str, dict[str, Any]] = {}
    for spec, label in zip(variants, labels, strict=True):
        records[label] = evaluate_spec(
            spec,
            X_fit,
            y_fit,
            X_valid,
            y_valid,
            log=log,
            stage=f"ablation:{name}",
            description=label,
            **eval_kwargs,
        )
    best_label = min(records, key=lambda k: records[k][f"val_{SELECTION_METRIC}"])
    gain = incumbent[f"val_{SELECTION_METRIC}"] - records[best_label][f"val_{SELECTION_METRIC}"]
    adopted = gain >= min_gain
    summary = {
        "incumbent_trial": incumbent["trial_id"],
        "incumbent_value": incumbent[f"val_{SELECTION_METRIC}"],
        "candidates": {
            label: {
                "trial_id": r["trial_id"],
                f"val_{SELECTION_METRIC}": r[f"val_{SELECTION_METRIC}"],
                "val_roc_auc": r["val_roc_auc"],
                "val_average_precision": r["val_average_precision"],
                "val_expected_calibration_error": r["val_expected_calibration_error"],
                "n_features": r["n_features"],
                "n_trees": r["n_trees"],
                "fit_seconds": r["fit_seconds"],
            }
            for label, r in records.items()
        },
        "best_candidate": best_label,
        "gain_in_val_log_loss": gain,
        "min_gain_to_adopt": min_gain,
        "adopted": best_label if adopted else None,
    }
    if adopted:
        chosen_spec = variants[list(labels).index(best_label)]
        return chosen_spec, records[best_label], summary
    return incumbent_spec, incumbent, summary


# --------------------------------------------------------------------------- #
# Stage 1: selection on the frozen training portion only
# --------------------------------------------------------------------------- #
@dataclass
class SelectionConfig:
    valid_size: float = config.VALIDATION_SIZE
    random_state: int = config.RANDOM_STATE
    libraries: tuple[str, ...] = LIBRARIES
    search_axes: dict[str, list[tuple[str, list[Any]]]] = field(
        default_factory=lambda: {k: list(v) for k, v in SEARCH_AXES.items()}
    )
    param_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS
    heavy_tail_treatments: tuple[HeavyTail, ...] = ("log1p", "winsorize")
    indicator_variants: tuple[MissingIndicators, ...] = ("all", "deduplicated")
    ablation_min_gain: float = 5e-4
    n_calibration_bins: int = 10
    fit_reference: bool = True
    lr_max_iter: int = 1000

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["search_axes"] = {
            lib: [[axis, list(values)] for axis, values in axes]
            for lib, axes in self.search_axes.items()
        }
        return out


def run_selection_stage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: SelectionConfig | None = None,
    *,
    log: TrialLog | None = None,
) -> dict[str, Any]:
    """Internal model selection. Receives the frozen *training* portion only.

    The signature is the leakage guarantee: no test rows can reach this function.
    """
    cfg = cfg or SelectionConfig()
    log = log if log is not None else TrialLog()
    started = time.perf_counter()

    X_fit, X_valid, y_fit, y_valid = make_split(
        X_train, y_train, test_size=cfg.valid_size, random_state=cfg.random_state
    )
    logger.info(
        "Validation split inside the training portion: fit=%d (prev %.4f) | valid=%d (prev %.4f)",
        len(y_fit),
        y_fit.mean(),
        len(y_valid),
        y_valid.mean(),
    )
    eval_kwargs = {
        "early_stopping_rounds": cfg.early_stopping_rounds,
        "n_bins": cfg.n_calibration_bins,
    }

    reference: dict[str, Any] = {}
    cached_ref = log.find_id("logistic_regression-ref")
    if cfg.fit_reference and cached_ref is not None:
        log.reused.append("logistic_regression-ref")
        reference["logistic_regression"] = {
            "fit_seconds": cached_ref["fit_seconds"],
            "n_features": int(cached_ref["n_features"]),
            "converged": cached_ref.get("converged"),
            "validation_metrics": {k: cached_ref[f"val_{k}"] for k in VAL_METRIC_KEYS},
        }
        logger.info("Reference LR on validation: reusing recorded trial")
    elif cfg.fit_reference:
        t0 = time.perf_counter()
        lr = build_baseline_pipeline(
            infer_feature_types(X_fit), max_iter=cfg.lr_max_iter, random_state=cfg.random_state
        ).fit(X_fit, y_fit)
        m_valid = compute_metrics(
            y_valid, lr.predict_proba(X_valid)[:, 1], n_bins=cfg.n_calibration_bins
        )
        reference["logistic_regression"] = {
            "fit_seconds": round(time.perf_counter() - t0, 2),
            "n_features": int(lr["preprocess"].get_feature_names_out().shape[0]),
            "converged": bool(np.asarray(lr["model"].n_iter_).max() < cfg.lr_max_iter),
            "validation_metrics": m_valid,
        }
        log.append(
            {
                "trial_id": "logistic_regression-ref",
                "library": "logistic_regression",
                "stage": "reference",
                "description": "Milestone 1 baseline pipeline refitted on the fit subset",
                "heavy_tail": "none",
                "missing_indicators": "all",
                "n_features": reference["logistic_regression"]["n_features"],
                "n_trees": None,
                "early_stopped": None,
                "converged": reference["logistic_regression"]["converged"],
                "fit_seconds": reference["logistic_regression"]["fit_seconds"],
                **_flatten_metrics("val", m_valid, VAL_METRIC_KEYS),
                **_flatten_metrics(
                    "train",
                    compute_metrics(y_fit, lr.predict_proba(X_fit)[:, 1]),
                    TRAIN_METRIC_KEYS,
                ),
                "params_json": json.dumps({"C": 1.0, "solver": "lbfgs", "penalty": "l2"}),
            }
        )
        del lr
        logger.info(
            "Reference LR on validation: log loss %.5f | ROC-AUC %.4f",
            m_valid["log_loss"],
            m_valid["roc_auc"],
        )

    libraries: dict[str, Any] = {}
    for library in cfg.libraries:
        base = ChallengerSpec.default(library, **cfg.param_overrides.get(library, {}))
        best_spec, best = coordinate_search(
            base,
            cfg.search_axes.get(library, []),
            X_fit,
            y_fit,
            X_valid,
            y_valid,
            log=log,
            **eval_kwargs,
        )
        search_best_trial = best["trial_id"]

        ablations: dict[str, Any] = {}
        if cfg.heavy_tail_treatments:
            variants = [
                best_spec.with_preprocessing(heavy_tail=t) for t in cfg.heavy_tail_treatments
            ]
            best_spec, best, ablations["heavy_tail"] = _ablation(
                "heavy_tail",
                best_spec,
                best,
                variants,
                list(cfg.heavy_tail_treatments),
                X_fit,
                y_fit,
                X_valid,
                y_valid,
                log=log,
                min_gain=cfg.ablation_min_gain,
                **eval_kwargs,
            )
        if cfg.indicator_variants:
            variants = [
                best_spec.with_preprocessing(missing_indicators=v) for v in cfg.indicator_variants
            ]
            best_spec, best, ablations["missing_indicators"] = _ablation(
                "missing_indicators",
                best_spec,
                best,
                variants,
                list(cfg.indicator_variants),
                X_fit,
                y_fit,
                X_valid,
                y_valid,
                log=log,
                min_gain=cfg.ablation_min_gain,
                **eval_kwargs,
            )

        locked = best_spec.with_params(n_estimators=int(best["n_trees"]))
        libraries[library] = {
            "search_best_trial": search_best_trial,
            "selected_trial": best["trial_id"],
            "n_trials": sum(1 for r in log.records if r["library"] == library),
            "ablations": ablations,
            "locked_spec": locked.to_dict(),
            "locked_n_estimators": int(best["n_trees"]),
            "validation_metrics": {k: best[f"val_{k}"] for k in VAL_METRIC_KEYS},
            "fit_metrics": {k: best[f"train_{k}"] for k in TRAIN_METRIC_KEYS},
            "fit_seconds": best["fit_seconds"],
            "n_features": best["n_features"],
        }
        logger.info(
            "%s locked: trial %s, %d rounds, val log loss %.5f, ROC-AUC %.4f",
            library,
            best["trial_id"],
            best["n_trees"],
            best["val_log_loss"],
            best["val_roc_auc"],
        )

    return {
        "run": _run_info(time.perf_counter() - started),
        "config": cfg.to_dict(),
        "selection_metric": f"validation {SELECTION_METRIC} (lower is better); "
        "ROC-AUC, PR-AUC, Brier and ECE recorded for every trial",
        "validation_split": {
            "strategy": "stratified random holdout carved out of the frozen training portion",
            "valid_size": cfg.valid_size,
            "random_state": cfg.random_state,
            "n_fit": int(len(y_fit)),
            "n_valid": int(len(y_valid)),
            "prevalence_fit": float(y_fit.mean()),
            "prevalence_valid": float(y_valid.mean()),
        },
        "reference": reference,
        "libraries": libraries,
        "n_trials_total": len(log.records),
        "n_trials_reused_from_log": len(log.reused),
    }


# --------------------------------------------------------------------------- #
# Stage 2: locked configurations on the frozen test partition
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentPaths:
    data_path: Path = config.APPLICATION_TRAIN_PATH
    membership_path: Path = config.PROCESSED_DATA_DIR / "baseline_split_membership.csv"
    baseline_metrics_path: Path = config.METRICS_DIR / "baseline_metrics.json"
    baseline_model_path: Path = config.MODELS_DIR / "baseline_logistic_regression.joblib"
    figures_dir: Path = config.FIGURES_DIR
    metrics_dir: Path = config.METRICS_DIR
    models_dir: Path = config.MODELS_DIR
    processed_dir: Path = config.PROCESSED_DATA_DIR

    @property
    def trials_path(self) -> Path:
        return self.metrics_dir / "challenger_trials.csv"

    @property
    def selection_path(self) -> Path:
        return self.metrics_dir / "challenger_selection.json"

    @property
    def comparison_path(self) -> Path:
        return self.metrics_dir / "model_comparison.csv"

    @property
    def bootstrap_path(self) -> Path:
        return self.metrics_dir / "bootstrap_comparison.json"

    @property
    def predictions_path(self) -> Path:
        return self.processed_dir / "challenger_test_predictions.csv"

    def metrics_path(self, library: str) -> Path:
        return self.metrics_dir / f"{library}_metrics.json"

    def model_path(self, library: str) -> Path:
        return self.models_dir / f"{library}_challenger.joblib"

    def to_dict(self) -> dict[str, Any]:
        return {k: _display_path(v) for k, v in asdict(self).items()}


@dataclass
class FinalConfig:
    n_bootstrap: int = 1000
    bootstrap_seed: int = config.RANDOM_STATE
    n_calibration_bins: int = 10
    save_models: bool = True
    save_predictions: bool = True


def _run_info(seconds: float) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "riskpilot_version": __version__,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
        "xgboost_version": xgb.__version__,
        "total_seconds": round(seconds, 1),
    }


def run_final_stage(
    split: comparison.FrozenSplit,
    selection: Mapping[str, Any],
    paths: ExperimentPaths,
    cfg: FinalConfig | None = None,
) -> dict[str, Any]:
    """Refit the locked specs on the whole training portion; score the test split once."""
    cfg = cfg or FinalConfig()
    started = time.perf_counter()
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)

    baseline = comparison.load_baseline_reference(
        split, model_path=paths.baseline_model_path, metrics_path=paths.baseline_metrics_path
    )
    predictions: dict[str, np.ndarray] = {"logistic_regression": baseline.y_prob_test}
    summaries: dict[str, dict[str, Any]] = {"logistic_regression": baseline.summary}
    payloads: dict[str, dict[str, Any]] = {}

    for library, entry in selection["libraries"].items():
        spec = ChallengerSpec.from_dict(entry["locked_spec"])
        logger.info(
            "Final %s: refitting locked spec (%d rounds) on %d rows",
            library,
            spec.params["n_estimators"],
            len(split.y_train),
        )
        result = fit_challenger(
            spec, split.X_train, split.y_train
        )  # no validation set: no early stopping
        t0 = time.perf_counter()
        p_test = result.pipeline.predict_proba(split.X_test)[:, 1]
        predict_seconds = time.perf_counter() - t0
        p_train = result.pipeline.predict_proba(split.X_train)[:, 1]
        m_test = compute_metrics(split.y_test, p_test, n_bins=cfg.n_calibration_bins)
        m_train = compute_metrics(split.y_train, p_train, n_bins=cfg.n_calibration_bins)
        table = calibration_table(split.y_test, p_test, n_bins=cfg.n_calibration_bins)
        figures = make_evaluation_figures(
            split.y_test,
            p_test,
            figures_dir=paths.figures_dir,
            prefix=library,
            n_bins=cfg.n_calibration_bins,
        )

        model_path: Path | None = None
        model_size_mb: float | None = None
        if cfg.save_models:
            paths.models_dir.mkdir(parents=True, exist_ok=True)
            model_path = paths.model_path(library)
            joblib.dump(result.pipeline, model_path)
            model_size_mb = round(model_path.stat().st_size / 1e6, 3)

        model = result.model
        payload: dict[str, Any] = {
            "run": _run_info(time.perf_counter() - started),
            "library": library,
            "label": LIBRARY_LABELS[library],
            "spec": spec.to_dict(),
            "selection": {
                "selected_trial": entry["selected_trial"],
                "validation_metrics": entry["validation_metrics"],
                "locked_n_estimators": entry["locked_n_estimators"],
                "selection_file": _display_path(paths.selection_path),
            },
            "data": {
                "path": _display_path(paths.data_path),
                "n_rows": int(len(split.y_train) + len(split.y_test)),
                "n_raw_features": int(split.X_train.shape[1]),
                "preprocessing": result.preprocessor.describe(),
            },
            "split": split.summary(),
            "model": {
                "estimator": type(model).__name__,
                "library_version": lgb.__version__ if library == "lightgbm" else xgb.__version__,
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
            "artifacts": {
                "figures": {k: _display_path(v) for k, v in figures.items()},
                "model": _display_path(model_path) if model_path else None,
            },
        }
        save_metrics(payload, paths.metrics_path(library))
        payloads[library] = payload
        predictions[library] = p_test
        summaries[library] = {
            "label": LIBRARY_LABELS[library],
            "n_features": result.n_features,
            "n_trees": result.n_trees,
            "fit_seconds": round(result.fit_seconds, 2),
            "predict_seconds_test": round(predict_seconds, 3),
            "model_size_mb": model_size_mb,
            "metrics_test": m_test,
            "metrics_train": m_train,
        }
        logger.info(
            "Final %s on the frozen test split: ROC-AUC %.4f | AP %.4f | log loss %.5f | "
            "Brier %.5f | ECE %.4f",
            library,
            m_test["roc_auc"],
            m_test["average_precision"],
            m_test["log_loss"],
            m_test["brier_score"],
            m_test["expected_calibration_error"],
        )
        del result, p_train

    table_df = comparison.comparison_table(summaries)
    table_df.to_csv(paths.comparison_path, index=False)

    boot = comparison.paired_bootstrap(
        split.y_test,
        predictions,
        n_replicates=cfg.n_bootstrap,
        seed=cfg.bootstrap_seed,
        reference="logistic_regression",
    )
    boot["run"] = _run_info(time.perf_counter() - started)
    save_metrics(boot, paths.bootstrap_path)

    figures = comparison.make_comparison_figures(
        split.y_test,
        predictions,
        boot,
        figures_dir=paths.figures_dir,
        n_bins=cfg.n_calibration_bins,
    )

    predictions_path: Path | None = None
    if cfg.save_predictions:
        paths.processed_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = paths.predictions_path
        out = pd.DataFrame(
            {split.ids_test.name or "row": split.ids_test.to_numpy(), "y": split.y_test.to_numpy()}
        )
        for name, p in predictions.items():
            out[f"p_{name}"] = p
        out.to_csv(predictions_path, index=False)

    return {
        "run": _run_info(time.perf_counter() - started),
        "baseline_integrity": baseline.integrity,
        "summaries": summaries,
        "comparison_table": table_df,
        "bootstrap": boot,
        "payloads": payloads,
        "predictions": predictions,
        "artifacts": {
            "comparison_table": _display_path(paths.comparison_path),
            "bootstrap": _display_path(paths.bootstrap_path),
            "figures": {k: _display_path(v) for k, v in figures.items()},
            "predictions": _display_path(predictions_path) if predictions_path else None,
            "metrics": {lib: _display_path(paths.metrics_path(lib)) for lib in payloads},
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_frozen_split(
    paths: ExperimentPaths, *, nrows: int | None = None
) -> comparison.FrozenSplit:
    """Load the table and rebuild the Milestone 1 split from the membership file.

    String columns are stored as categoricals to halve memory; the baseline
    integrity check in :func:`run_final_stage` confirms the persisted Logistic
    Regression pipeline reproduces its recorded metrics on that representation.
    """
    df = load_application_train(paths.data_path, nrows=nrows)
    validate_target(df[config.TARGET_COL])
    X, y = split_features_target(categorize_strings(df))  # halves the table's memory footprint
    ids = df[config.ID_COL]
    del df
    return comparison.load_frozen_split(X, y, ids, membership_path=paths.membership_path)


def run_experiment(
    paths: ExperimentPaths,
    *,
    stage: str = "all",
    nrows: int | None = None,
    selection_cfg: SelectionConfig | None = None,
    final_cfg: FinalConfig | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    split = load_frozen_split(paths, nrows=nrows)
    out: dict[str, Any] = {"split": split.summary()}

    if stage in ("all", "select"):
        log = TrialLog(paths.trials_path, resume=resume)
        selection = run_selection_stage(split.X_train, split.y_train, selection_cfg, log=log)
        selection["artifacts"] = {"trials": _display_path(paths.trials_path)}
        save_metrics(selection, paths.selection_path)
        out["selection"] = selection
        logger.info(
            "Saved selection to %s (%d trials)",
            _display_path(paths.selection_path),
            len(log.records),
        )
    if stage in ("all", "final"):
        selection = out.get("selection") or json.loads(
            paths.selection_path.read_text(encoding="utf-8")
        )
        out["final"] = run_final_stage(split, selection, paths, final_cfg)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riskpilot.models.challengers",
        description="Gradient-boosting challengers vs. the frozen Logistic Regression baseline.",
    )
    parser.add_argument("--stage", choices=["all", "select", "final"], default="all")
    parser.add_argument("--data-path", type=Path, default=config.APPLICATION_TRAIN_PATH)
    parser.add_argument("--nrows", type=int, default=None, help="Row limit for smoke runs.")
    parser.add_argument(
        "--quick", action="store_true", help="Tiny search grid and few boosting rounds."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse trials already recorded in the trial log instead of refitting them.",
    )
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--membership-path", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=config.FIGURES_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=config.METRICS_DIR)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DATA_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    use_headless_backend()
    args = _build_parser().parse_args(argv)
    paths = ExperimentPaths(
        data_path=args.data_path,
        figures_dir=args.figures_dir,
        metrics_dir=args.metrics_dir,
        models_dir=args.models_dir,
        processed_dir=args.processed_dir,
    )
    paths.membership_path = args.membership_path or (
        paths.processed_dir / "baseline_split_membership.csv"
    )
    paths.baseline_metrics_path = paths.metrics_dir / "baseline_metrics.json"
    paths.baseline_model_path = paths.models_dir / "baseline_logistic_regression.joblib"

    overrides = {lib: {"n_jobs": args.n_jobs} for lib in LIBRARIES}
    selection_cfg = SelectionConfig(param_overrides=overrides)
    if args.quick:
        selection_cfg = SelectionConfig(
            search_axes={k: list(v) for k, v in SMOKE_SEARCH_AXES.items()},
            param_overrides={
                lib: {**SMOKE_PARAM_OVERRIDES[lib], "n_jobs": args.n_jobs} for lib in LIBRARIES
            },
            early_stopping_rounds=10,
            lr_max_iter=500,
        )
    final_cfg = FinalConfig(n_bootstrap=args.n_bootstrap, save_models=not args.no_save_models)

    try:
        out = run_experiment(
            paths,
            stage=args.stage,
            nrows=args.nrows,
            selection_cfg=selection_cfg,
            final_cfg=final_cfg,
            resume=args.resume,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    if "final" in out:
        table = out["final"]["comparison_table"]
        cols = [
            "model",
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
        print(f"\nBaseline integrity: {out['final']['baseline_integrity']}")
        print(f"Artifacts: {json.dumps(out['final']['artifacts'], indent=1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
