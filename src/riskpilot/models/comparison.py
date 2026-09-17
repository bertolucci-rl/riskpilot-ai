"""Frozen-split handling, baseline reference, model comparison and paired bootstrap.

Everything that touches the frozen Milestone 1 test partition for the challenger
experiment goes through this module:

* :func:`load_frozen_split` rebuilds the train / test partition from the persisted
  membership file and verifies it against a re-derivation from the seed;
* :func:`load_baseline_reference` reloads the persisted Logistic Regression
  pipeline, scores the test partition and checks the result against the metrics
  recorded in Milestone 1 (baseline integrity);
* :func:`comparison_table` and :func:`paired_bootstrap` compare final models on
  identical test rows; the bootstrap resamples the *same* observations for every
  model so the uncertainty of a *difference* is estimated, not of two independent
  numbers;
* the ``plot_*`` functions draw the comparison figures.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from riskpilot import config
from riskpilot.features.preprocessing import build_baseline_pipeline, infer_feature_types
from riskpilot.models.evaluate import (
    MODEL_COLOR,
    REFERENCE_COLOR,
    SECOND_COLOR,
    _as_arrays,
    _finish,
    _style_axes,
    calibration_table,
    compute_metrics,
)
from riskpilot.models.train import make_split

logger = logging.getLogger(__name__)

__all__ = [
    "BOOTSTRAP_METRICS",
    "SERIES_COLORS",
    "BaselineReference",
    "FrozenSplit",
    "bootstrap_replicates",
    "comparison_table",
    "load_baseline_reference",
    "load_frozen_split",
    "make_comparison_figures",
    "paired_bootstrap",
    "plot_calibration_comparison",
    "plot_delta_bootstrap",
    "plot_precision_recall_comparison",
    "plot_probability_distributions",
    "plot_roc_comparison",
]

# Categorical slots 1-3 of the project palette, in fixed order (validated all-pairs).
SERIES_COLORS: dict[str, str] = {
    "logistic_regression": MODEL_COLOR,  # blue
    "lightgbm": SECOND_COLOR,  # orange
    "xgboost": "#1baf7a",  # aqua
}
LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}
BOOTSTRAP_METRICS: tuple[str, ...] = ("roc_auc", "average_precision", "log_loss", "brier_score")
HIGHER_IS_BETTER: dict[str, bool] = {
    "roc_auc": True,
    "average_precision": True,
    "log_loss": False,
    "brier_score": False,
}
METRIC_LABELS: dict[str, str] = {
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC (average precision)",
    "log_loss": "Log loss",
    "brier_score": "Brier score",
}


def _label(name: str) -> str:
    return LABELS.get(name, name)


def _color(name: str, index: int) -> str:
    if name in SERIES_COLORS:
        return SERIES_COLORS[name]
    fallback = list(SERIES_COLORS.values())
    return fallback[index % len(fallback)]


# --------------------------------------------------------------------------- #
# Frozen split
# --------------------------------------------------------------------------- #
@dataclass
class FrozenSplit:
    """The Milestone 1 partition, rebuilt from the persisted membership file."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series
    membership_path: Path
    verified_against_seed: bool

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": "frozen Milestone 1 stratified holdout, reloaded from the membership file",
            "membership_file": self.membership_path.as_posix(),
            "verified_against_reseeded_split": self.verified_against_seed,
            "n_train": int(len(self.y_train)),
            "n_test": int(len(self.y_test)),
            "n_test_positive": int(self.y_test.sum()),
            "prevalence_train": float(self.y_train.mean()),
            "prevalence_test": float(self.y_test.mean()),
        }


def load_frozen_split(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    *,
    membership_path: str | Path,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    verify: bool = True,
) -> FrozenSplit:
    """Partition ``X, y`` exactly as the persisted membership file says.

    ``ids`` must cover the same identifiers as the membership file (no more, no
    less). When ``verify`` is true the split is also re-derived from the seed
    and must match, otherwise a ``ValueError`` is raised: silently modelling on
    a different holdout would invalidate every comparison with Milestone 1.
    """
    path = Path(membership_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split membership file not found at '{path}'. Run the Milestone 1 baseline "
            "(python -m riskpilot.models.train) first; it persists the frozen split."
        )
    membership = pd.read_csv(path)
    id_col = ids.name or "row"
    if id_col not in membership.columns or "split" not in membership.columns:
        raise ValueError(
            f"Membership file must have columns ['{id_col}', 'split'], "
            f"got {list(membership.columns)}."
        )
    if membership[id_col].duplicated().any():
        raise ValueError("Membership file contains duplicated identifiers.")
    unexpected = set(membership["split"].unique()) - {"train", "test"}
    if unexpected:
        raise ValueError(f"Membership file has unexpected split labels: {sorted(unexpected)}.")
    if set(membership[id_col]) != set(ids):
        raise ValueError(
            "Identifiers in the data do not match the membership file "
            f"({len(set(ids))} in data vs {membership[id_col].nunique()} in file). "
            "The challengers must be evaluated on the exact Milestone 1 rows."
        )
    split = membership.set_index(id_col)["split"].reindex(ids.to_numpy()).to_numpy()
    test_mask = split == "test"

    verified = False
    if verify:
        _, X_test_again, _, _ = make_split(X, y, test_size=test_size, random_state=random_state)
        verified = set(ids.loc[X_test_again.index]) == set(ids[test_mask])
        if not verified:
            raise ValueError(
                "Re-deriving the split from the seed does not reproduce the membership file; "
                "refusing to evaluate on a holdout that differs from Milestone 1."
            )

    return FrozenSplit(
        X_train=X.loc[~test_mask],
        y_train=y.loc[~test_mask],
        X_test=X.loc[test_mask],
        y_test=y.loc[test_mask],
        ids_train=ids.loc[~test_mask],
        ids_test=ids.loc[test_mask],
        membership_path=path,
        verified_against_seed=verified,
    )


# --------------------------------------------------------------------------- #
# Baseline reference
# --------------------------------------------------------------------------- #
@dataclass
class BaselineReference:
    y_prob_test: np.ndarray
    summary: dict[str, Any]
    integrity: dict[str, Any]


def load_baseline_reference(
    split: FrozenSplit,
    *,
    model_path: str | Path,
    metrics_path: str | Path,
    n_bins: int = 10,
    atol_reloaded: float = 1e-6,
    atol_refit: float = 1e-3,
) -> BaselineReference:
    """Score the frozen test partition with the Milestone 1 pipeline and check integrity.

    The persisted pipeline is preferred (no fitting, bit-for-bit the model that
    produced ``baseline_metrics.json``). If it is absent the baseline is refitted
    on the training portion with the recorded configuration; that reproduces the
    metrics only up to optimizer tolerance, so a looser check applies and the
    fact is recorded.
    """
    model_path, metrics_path = Path(model_path), Path(metrics_path)
    recorded: dict[str, Any] | None = None
    if metrics_path.is_file():
        recorded = json.loads(metrics_path.read_text(encoding="utf-8"))

    reloaded = model_path.is_file()
    if reloaded:
        pipeline = joblib.load(model_path)
        fit_seconds = recorded["model"]["fit_seconds"] if recorded else None
    else:
        logger.warning(
            "Baseline model %s not found; refitting the baseline on the training portion.",
            model_path,
        )
        cfg = (recorded or {}).get("config", {})
        t0 = time.perf_counter()
        pipeline = build_baseline_pipeline(
            infer_feature_types(split.X_train),
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            random_state=cfg.get("random_state", config.RANDOM_STATE),
        ).fit(split.X_train, split.y_train)
        fit_seconds = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    p_test = pipeline.predict_proba(split.X_test)[:, 1]
    predict_seconds = time.perf_counter() - t0
    m_test = compute_metrics(split.y_test, p_test, n_bins=n_bins)

    integrity: dict[str, Any] = {
        "model_reloaded_from_disk": reloaded,
        "model_path": model_path.as_posix(),
        "recorded_metrics_available": recorded is not None,
        "split_verified_against_reseeded_split": split.verified_against_seed,
    }
    if recorded is not None:
        rec_test = recorded["metrics"]["test"]
        checks = {
            k: abs(m_test[k] - rec_test[k])
            for k in ("roc_auc", "average_precision", "log_loss", "brier_score")
        }
        tol = atol_reloaded if reloaded else atol_refit
        integrity.update(
            {
                "n_test_matches": int(rec_test["n"]) == int(len(split.y_test)),
                "max_abs_metric_difference": max(checks.values()),
                "tolerance": tol,
                "metrics_match_recorded": bool(
                    int(rec_test["n"]) == int(len(split.y_test)) and max(checks.values()) <= tol
                ),
                "recorded_test_metrics": {k: rec_test[k] for k in checks},
            }
        )
        if not integrity["metrics_match_recorded"]:
            raise ValueError(
                "The baseline scored on the frozen test split does not reproduce the recorded "
                f"Milestone 1 metrics (max |diff| = {max(checks.values()):.2e} > {tol}); "
                "refusing to compare challengers against a baseline that has drifted."
            )

    if recorded is not None:
        m_train = recorded["metrics"]["train"]
    else:
        m_train = compute_metrics(
            split.y_train, pipeline.predict_proba(split.X_train)[:, 1], n_bins=n_bins
        )

    summary = {
        "label": _label("logistic_regression"),
        "n_features": int(pipeline["preprocess"].get_feature_names_out().shape[0]),
        "n_trees": None,
        "fit_seconds": fit_seconds,
        "predict_seconds_test": round(predict_seconds, 3),
        "model_size_mb": round(model_path.stat().st_size / 1e6, 3) if reloaded else None,
        "metrics_test": m_test,
        "metrics_train": m_train,
    }
    return BaselineReference(y_prob_test=p_test, summary=summary, integrity=integrity)


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
TABLE_METRICS: tuple[str, ...] = (
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "brier_skill_score",
    "expected_calibration_error",
    "calibration_slope",
    "calibration_intercept",
    "mean_predicted_probability",
    "max_predicted_probability",
)


def comparison_table(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    reference: str = "logistic_regression",
) -> pd.DataFrame:
    """One row per model with test metrics, cost figures and deltas vs. the reference.

    Deltas are ``model - reference`` (absolute); ``rel_*`` columns are the same
    differences as a percentage of the reference value.
    """
    if reference not in summaries:
        raise KeyError(f"Reference model '{reference}' is not among {list(summaries)}.")
    ref = summaries[reference]["metrics_test"]
    rows = []
    for name, s in summaries.items():
        m = s["metrics_test"]
        row: dict[str, Any] = {"model": s.get("label", _label(name)), "key": name}
        row.update({k: m.get(k) for k in TABLE_METRICS})
        row.update(
            {
                "train_roc_auc": s["metrics_train"].get("roc_auc"),
                "train_log_loss": s["metrics_train"].get("log_loss"),
                "n_features": s.get("n_features"),
                "n_trees": s.get("n_trees"),
                "fit_seconds": s.get("fit_seconds"),
                "predict_seconds_test": s.get("predict_seconds_test"),
                "model_size_mb": s.get("model_size_mb"),
            }
        )
        for k in BOOTSTRAP_METRICS:
            delta = m[k] - ref[k]
            row[f"delta_{k}"] = delta
            row[f"rel_{k}_pct"] = 100.0 * delta / ref[k] if ref[k] else float("nan")
        row["delta_expected_calibration_error"] = (
            m["expected_calibration_error"] - ref["expected_calibration_error"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #
def _weighted_metrics(
    y: np.ndarray, p: np.ndarray, w: np.ndarray, metrics: Sequence[str]
) -> list[float]:
    out = []
    for m in metrics:
        if m == "roc_auc":
            out.append(float(roc_auc_score(y, p, sample_weight=w)))
        elif m == "average_precision":
            out.append(float(average_precision_score(y, p, sample_weight=w)))
        elif m == "log_loss":
            out.append(float(log_loss(y, p, sample_weight=w, labels=[0, 1])))
        elif m == "brier_score":
            out.append(float(brier_score_loss(y, p, sample_weight=w)))
        else:
            raise ValueError(f"Unsupported bootstrap metric '{m}'.")
    return out


def bootstrap_replicates(
    y_true: Any,
    predictions: Mapping[str, Any],
    *,
    n_replicates: int = 1000,
    seed: int = config.RANDOM_STATE,
    metrics: Sequence[str] = BOOTSTRAP_METRICS,
) -> dict[str, pd.DataFrame]:
    """Metric values of every model on the *same* ``n_replicates`` resamples.

    Each replicate draws ``n`` observations with replacement; the draw is
    expressed as multinomial counts used as ``sample_weight`` so that every
    model is scored on identical resampled rows (paired design). Returns one
    ``DataFrame`` (replicates x metrics) per model.
    """
    if not predictions:
        raise ValueError("predictions must contain at least one model.")
    arrays: dict[str, np.ndarray] = {}
    y = None
    for name, p in predictions.items():
        y_i, p_i = _as_arrays(y_true, p)
        if y is None:
            y = y_i
        arrays[name] = p_i
    assert y is not None
    n = y.size
    if y.min() == y.max():
        raise ValueError("Bootstrap requires both classes in y_true.")
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive.")

    rng = np.random.default_rng(seed)
    values = {name: np.empty((n_replicates, len(metrics)), dtype=float) for name in arrays}
    for b in range(n_replicates):
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n).astype(float)
        if y[w > 0].min() == y[w > 0].max():  # pragma: no cover - only for tiny inputs
            w = np.ones(n)
        for name, p in arrays.items():
            values[name][b] = _weighted_metrics(y, p, w, metrics)
    return {name: pd.DataFrame(v, columns=list(metrics)) for name, v in values.items()}


def paired_bootstrap(
    y_true: Any,
    predictions: Mapping[str, Any],
    *,
    n_replicates: int = 1000,
    seed: int = config.RANDOM_STATE,
    reference: str = "logistic_regression",
    metrics: Sequence[str] = BOOTSTRAP_METRICS,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Percentile confidence intervals for each model's metrics and for differences.

    Differences are ``model - reference`` computed replicate by replicate on the
    same resampled observations. ``fraction_improving`` is the share of
    replicates in which the model beats the reference on that metric; it is a
    stability summary, not a significance test.
    """
    if reference not in predictions:
        raise KeyError(f"Reference '{reference}' not among predictions {list(predictions)}.")
    y, _ = _as_arrays(y_true, predictions[reference])
    reps = bootstrap_replicates(
        y, predictions, n_replicates=n_replicates, seed=seed, metrics=metrics
    )
    alpha = (1.0 - confidence_level) / 2.0
    ones = np.ones(y.size)

    def summarize(sample: np.ndarray, estimate: float) -> dict[str, float]:
        lo, hi = np.quantile(sample, [alpha, 1.0 - alpha])
        return {
            "estimate": float(estimate),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "bootstrap_mean": float(sample.mean()),
            "bootstrap_sd": float(sample.std(ddof=1)) if sample.size > 1 else 0.0,
        }

    models: dict[str, Any] = {}
    point: dict[str, dict[str, float]] = {}
    for name, p in predictions.items():
        p_arr = np.asarray(p, dtype=float)
        point[name] = dict(zip(metrics, _weighted_metrics(y, p_arr, ones, metrics), strict=True))
        models[name] = {m: summarize(reps[name][m].to_numpy(), point[name][m]) for m in metrics}

    differences: dict[str, Any] = {}
    names = list(predictions)
    pairs = [(n, reference) for n in names if n != reference]
    others = [n for n in names if n != reference]
    pairs += [(b, a) for i, a in enumerate(others) for b in others[i + 1 :]]
    for model, base in pairs:
        entry: dict[str, Any] = {"model": model, "reference": base}
        for m in metrics:
            diff = reps[model][m].to_numpy() - reps[base][m].to_numpy()
            s = summarize(diff, point[model][m] - point[base][m])
            better = diff > 0 if HIGHER_IS_BETTER[m] else diff < 0
            s["fraction_improving"] = float(better.mean())
            s["ci_excludes_zero"] = bool(s["ci_low"] > 0 or s["ci_high"] < 0)
            entry[m] = s
        differences[f"{model}_vs_{base}"] = entry

    return {
        "method": "paired percentile bootstrap; each replicate resamples the test rows with "
        "replacement (multinomial counts used as sample weights) and scores every model on the "
        "same resample",
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "n_replicates": int(n_replicates),
        "seed": int(seed),
        "confidence_level": confidence_level,
        "reference": reference,
        "metrics": list(metrics),
        "models": models,
        "differences": differences,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_roc_comparison(
    y_true: Any, predictions: Mapping[str, Any], *, path: str | Path | None = None
) -> Figure:
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linewidth=1, linestyle="--", label="Chance")
    for i, (name, p) in enumerate(predictions.items()):
        y, p_arr = _as_arrays(y_true, p)
        fpr, tpr, _ = roc_curve(y, p_arr)
        ax.plot(
            fpr,
            tpr,
            color=_color(name, i),
            linewidth=2,
            label=f"{_label(name)} (AUC = {roc_auc_score(y, p_arr):.3f})",
        )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (frozen test set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    _style_axes(ax)
    return _finish(fig, path)


def plot_precision_recall_comparison(
    y_true: Any, predictions: Mapping[str, Any], *, path: str | Path | None = None
) -> Figure:
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    first = next(iter(predictions.values()))
    y, _ = _as_arrays(y_true, first)
    prevalence = y.mean()
    ax.axhline(
        prevalence,
        color=REFERENCE_COLOR,
        linewidth=1,
        linestyle="--",
        label=f"Prevalence ({prevalence:.3f})",
    )
    for i, (name, p) in enumerate(predictions.items()):
        y, p_arr = _as_arrays(y_true, p)
        precision, recall, _ = precision_recall_curve(y, p_arr)
        ax.plot(
            recall,
            precision,
            color=_color(name, i),
            linewidth=2,
            label=f"{_label(name)} (AP = {average_precision_score(y, p_arr):.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (frozen test set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=False)
    _style_axes(ax)
    return _finish(fig, path)


def plot_calibration_comparison(
    y_true: Any,
    predictions: Mapping[str, Any],
    *,
    path: str | Path | None = None,
    n_bins: int = 10,
) -> Figure:
    """Reliability diagrams on quantile bins, one line per model, shared axes."""
    tables = {
        name: calibration_table(y_true, p, n_bins=n_bins, strategy="quantile")
        for name, p in predictions.items()
    }
    limit = (
        max(
            float(max(t["mean_predicted"].max(), t["observed_rate"].max())) for t in tables.values()
        )
        * 1.15
    )
    limit = min(max(limit, 0.05), 1.0)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot(
        [0, limit],
        [0, limit],
        color=REFERENCE_COLOR,
        linewidth=1,
        linestyle="--",
        label="Perfect calibration",
    )
    for i, (name, table) in enumerate(tables.items()):
        ax.plot(
            table["mean_predicted"],
            table["observed_rate"],
            color=_color(name, i),
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=_label(name),
        )
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed default rate (per bin)")
    ax.set_title(f"Reliability diagrams, {n_bins} quantile bins (frozen test set)")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.legend(loc="upper left", frameon=False)
    _style_axes(ax)
    return _finish(fig, path)


def plot_probability_distributions(
    y_true: Any,
    predictions: Mapping[str, Any],
    *,
    path: str | Path | None = None,
    bins: int = 50,
) -> Figure:
    """Small multiples: class-conditional predicted-probability densities per model."""
    names = list(predictions)
    upper = max(float(np.asarray(p, dtype=float).max()) for p in predictions.values())
    edges = np.linspace(0.0, max(upper, 1e-6), bins + 1)
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.4 * len(names), 4.0), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names, strict=True):
        y, p = _as_arrays(y_true, predictions[name])
        ax.hist(
            p[y == 0],
            bins=edges,
            density=True,
            histtype="step",
            linewidth=2,
            color=MODEL_COLOR,
            label="Non-default",
        )
        ax.hist(
            p[y == 1],
            bins=edges,
            density=True,
            histtype="step",
            linewidth=2,
            color=SECOND_COLOR,
            label="Default",
        )
        ax.axvline(
            float(y.mean()),
            color=REFERENCE_COLOR,
            linewidth=1,
            linestyle="--",
            label=f"Prevalence ({y.mean():.3f})",
        )
        ax.set_title(_label(name))
        ax.set_xlabel("Predicted probability of default")
        _style_axes(ax)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.suptitle("Predicted-probability distributions by true class (frozen test set)", y=1.02)
    return _finish(fig, path)


def plot_delta_bootstrap(
    bootstrap: Mapping[str, Any],
    *,
    path: str | Path | None = None,
    reference: str | None = None,
) -> Figure:
    """Dot-and-whisker chart of ``model - reference`` with bootstrap CIs, one panel per metric."""
    reference = reference or bootstrap["reference"]
    metrics = list(bootstrap["metrics"])
    entries = [e for k, e in bootstrap["differences"].items() if e["reference"] == reference]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.7 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)
    level = int(round(100 * bootstrap.get("confidence_level", 0.95)))
    for ax, metric in zip(axes, metrics, strict=True):
        ax.axvline(0.0, color=REFERENCE_COLOR, linewidth=1, linestyle="--")
        for j, e in enumerate(entries):
            s = e[metric]
            color = _color(e["model"], j + 1)
            ax.errorbar(
                s["estimate"],
                j,
                xerr=[[s["estimate"] - s["ci_low"]], [s["ci_high"] - s["estimate"]]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=4,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label=_label(e["model"]),
            )
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([_label(e["model"]) for e in entries])
        ax.set_ylim(-0.7, len(entries) - 0.3)
        ax.invert_yaxis()
        better = "higher is better" if HIGHER_IS_BETTER[metric] else "lower is better"
        ax.set_title(f"Δ {METRIC_LABELS[metric].split(' (')[0]}\n({better})", fontsize=10)
        ax.set_xlabel(f"model − {_label(reference)}")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # sparse ticks: labels never collide
        ax.grid(axis="y", visible=False)
        _style_axes(ax)
    fig.suptitle(
        f"Differences vs. {_label(reference)} with {level} % paired-bootstrap intervals "
        f"({bootstrap['n_replicates']:,} replicates)",
        y=1.04,
    )
    return _finish(fig, path)


def make_comparison_figures(
    y_true: Any,
    predictions: Mapping[str, Any],
    bootstrap: Mapping[str, Any] | None,
    *,
    figures_dir: str | Path,
    prefix: str = "challenger",
    n_bins: int = 10,
    close: bool = True,
) -> dict[str, Path]:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "roc_comparison": figures_dir / f"{prefix}_roc_comparison.png",
        "precision_recall_comparison": figures_dir / f"{prefix}_precision_recall_comparison.png",
        "calibration_comparison": figures_dir / f"{prefix}_calibration_comparison.png",
        "probability_distributions": figures_dir / f"{prefix}_probability_distributions.png",
    }
    figs = [
        plot_roc_comparison(y_true, predictions, path=outputs["roc_comparison"]),
        plot_precision_recall_comparison(
            y_true, predictions, path=outputs["precision_recall_comparison"]
        ),
        plot_calibration_comparison(
            y_true, predictions, path=outputs["calibration_comparison"], n_bins=n_bins
        ),
        plot_probability_distributions(
            y_true, predictions, path=outputs["probability_distributions"]
        ),
    ]
    if bootstrap is not None:
        outputs["delta_bootstrap"] = figures_dir / f"{prefix}_delta_bootstrap.png"
        figs.append(plot_delta_bootstrap(bootstrap, path=outputs["delta_bootstrap"]))
    if close:
        for fig in figs:
            plt.close(fig)
    return outputs
