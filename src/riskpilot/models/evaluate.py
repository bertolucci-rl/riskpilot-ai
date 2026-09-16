"""Probabilistic evaluation of predicted default probabilities.

The model's job is to estimate ``P(Y = 1 | X)``, so evaluation is split into:

* **ranking quality** - ROC-AUC and average precision (PR-AUC);
* **probability quality** - log loss and Brier score, each reported next to the
  score of a constant "predict the prevalence" model so the numbers are
  interpretable (the Brier skill score is that comparison in one number);
* **calibration** - a quantile-binned reliability table and the expected
  calibration error (ECE) derived from it.

Plots are deliberately minimal: one axis per figure, thin marks, a reference
line where one is meaningful, and a legend whenever two series are drawn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

MODEL_COLOR = "#2a78d6"  # categorical slot 1 (blue)
SECOND_COLOR = "#eb6834"  # categorical slot 2 (orange)
REFERENCE_COLOR = "#8a8984"  # neutral ink for baselines / diagonals
GRID_COLOR = "#e5e4e0"

__all__ = [
    "calibration_slope_intercept",
    "calibration_table",
    "compute_metrics",
    "expected_calibration_error",
    "make_evaluation_figures",
    "plot_calibration_curve",
    "plot_precision_recall_curve",
    "plot_probability_distribution",
    "plot_roc_curve",
    "save_metrics",
]


def _as_arrays(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"Shape mismatch: y_true {y.shape} vs y_prob {p.shape}.")
    if y.size == 0:
        raise ValueError("Cannot evaluate an empty set of predictions.")
    if np.any((p < 0.0) | (p > 1.0)) or np.any(np.isnan(p)):
        raise ValueError("Predicted probabilities must lie in [0, 1] and contain no NaN.")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y_true must be binary (0/1).")
    return y, p


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibration_table(
    y_true: Any,
    y_prob: Any,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Reliability table: per-bin mean predicted probability vs. observed rate.

    ``strategy="quantile"`` puts an equal number of predictions in each bin, which
    is the informative choice when most probabilities are concentrated in a
    narrow low range (as they are for a low-prevalence default model).
    """
    y, p = _as_arrays(y_true, y_prob)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'.")
    if len(edges) < 2:
        edges = np.array([p.min(), p.max() + 1e-12])
    bin_ids = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["gap"] = table["observed_rate"] - table["mean_predicted"]
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Count-weighted mean absolute gap between predicted and observed rates."""
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())


def calibration_slope_intercept(
    y_true: Any, y_prob: Any, *, eps: float = 1e-12
) -> tuple[float | None, float | None]:
    """Logistic calibration diagnostic (Cox, 1958): fit ``y ~ a + b * logit(p)``.

    ``b = 1`` and ``a = 0`` mean the probabilities are calibrated; ``b < 1``
    means they are over-confident (too spread out), ``b > 1`` under-confident;
    ``a`` is a global shift given the slope. The fit is (practically)
    unpenalized. This is a *diagnostic*: nothing here alters the predictions.

    Returns ``(None, None)`` when the quantity is undefined: a single class,
    constant predictions, or predictions that separate the classes perfectly
    (the maximum-likelihood slope is then infinite).
    """
    y, p = _as_arrays(y_true, y_prob)
    p = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    if y.min() == y.max() or logit.max() == logit.min():
        return None, None
    pos, neg = logit[y == 1], logit[y == 0]
    if pos.min() > neg.max() or pos.max() < neg.min():
        return None, None
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit.reshape(-1, 1), y)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: Any, y_prob: Any, *, n_bins: int = 10) -> dict[str, float | int | None]:
    """Ranking, probabilistic and calibration metrics with prevalence references."""
    y, p = _as_arrays(y_true, y_prob)
    prevalence = float(y.mean())
    constant = np.full_like(p, prevalence)

    brier = float(brier_score_loss(y, p))
    brier_ref = float(brier_score_loss(y, constant))
    ll = float(log_loss(y, p, labels=[0, 1]))
    ll_ref = float(log_loss(y, constant, labels=[0, 1]))
    table = calibration_table(y, p, n_bins=n_bins, strategy="quantile")
    slope, intercept = calibration_slope_intercept(y, p)

    return {
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "average_precision_prevalence_baseline": prevalence,
        "log_loss": ll,
        "log_loss_prevalence_baseline": ll_ref,
        "brier_score": brier,
        "brier_prevalence_baseline": brier_ref,
        "brier_skill_score": 1.0 - brier / brier_ref if brier_ref > 0 else float("nan"),
        "expected_calibration_error": expected_calibration_error(table),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_predicted_probability": float(p.mean()),
        "median_predicted_probability": float(np.median(p)),
        "max_predicted_probability": float(p.max()),
    }


def save_metrics(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON document, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)


def _finish(fig: Figure, path: str | Path | None) -> Figure:
    fig.tight_layout()
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
    return fig


def plot_roc_curve(
    y_true: Any, y_prob: Any, *, path: str | Path | None = None, label: str = "Logistic Regression"
) -> Figure:
    y, p = _as_arrays(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linewidth=1, linestyle="--", label="Chance")
    ax.plot(fpr, tpr, color=MODEL_COLOR, linewidth=2, label=f"{label} (AUC = {auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    _style_axes(ax)
    return _finish(fig, path)


def plot_precision_recall_curve(
    y_true: Any, y_prob: Any, *, path: str | Path | None = None, label: str = "Logistic Regression"
) -> Figure:
    y, p = _as_arrays(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    prevalence = y.mean()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.axhline(
        prevalence,
        color=REFERENCE_COLOR,
        linewidth=1,
        linestyle="--",
        label=f"Prevalence ({prevalence:.3f})",
    )
    ax.plot(recall, precision, color=MODEL_COLOR, linewidth=2, label=f"{label} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=False)
    _style_axes(ax)
    return _finish(fig, path)


def plot_calibration_curve(
    y_true: Any,
    y_prob: Any,
    *,
    path: str | Path | None = None,
    n_bins: int = 10,
    label: str = "Logistic Regression",
) -> Figure:
    """Reliability diagram on quantile bins, zoomed to the observed probability range."""
    y, p = _as_arrays(y_true, y_prob)
    table = calibration_table(y, p, n_bins=n_bins, strategy="quantile")
    limit = float(max(table["mean_predicted"].max(), table["observed_rate"].max()) * 1.15)
    limit = min(max(limit, 0.05), 1.0)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(
        [0, limit],
        [0, limit],
        color=REFERENCE_COLOR,
        linewidth=1,
        linestyle="--",
        label="Perfect calibration",
    )
    ax.plot(
        table["mean_predicted"],
        table["observed_rate"],
        color=MODEL_COLOR,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label=f"{label} ({n_bins} quantile bins)",
    )
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed default rate (per bin)")
    ax.set_title("Reliability diagram (test set)")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.legend(loc="upper left", frameon=False)
    _style_axes(ax)
    return _finish(fig, path)


def plot_probability_distribution(
    y_true: Any, y_prob: Any, *, path: str | Path | None = None, bins: int = 50
) -> Figure:
    """Histogram of predicted probabilities, split by true class (density-normalized)."""
    y, p = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, max(float(p.max()), 1e-6), bins + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(
        p[y == 0],
        bins=edges,
        density=True,
        histtype="step",
        linewidth=2,
        color=MODEL_COLOR,
        label=f"Non-default (n = {int((y == 0).sum()):,})",
    )
    ax.hist(
        p[y == 1],
        bins=edges,
        density=True,
        histtype="step",
        linewidth=2,
        color=SECOND_COLOR,
        label=f"Default (n = {int((y == 1).sum()):,})",
    )
    ax.axvline(
        float(y.mean()),
        color=REFERENCE_COLOR,
        linewidth=1,
        linestyle="--",
        label=f"Prevalence ({y.mean():.3f})",
    )
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Density")
    ax.set_title("Predicted-probability distribution by true class (test set)")
    ax.legend(frameon=False)
    _style_axes(ax)
    return _finish(fig, path)


def make_evaluation_figures(
    y_true: Any,
    y_prob: Any,
    *,
    figures_dir: str | Path,
    prefix: str = "baseline",
    n_bins: int = 10,
    close: bool = True,
) -> dict[str, Path]:
    """Render and save the four standard figures; return their paths."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "roc_curve": figures_dir / f"{prefix}_roc_curve.png",
        "precision_recall_curve": figures_dir / f"{prefix}_precision_recall_curve.png",
        "calibration_curve": figures_dir / f"{prefix}_calibration_curve.png",
        "probability_distribution": figures_dir / f"{prefix}_probability_distribution.png",
    }
    figs = [
        plot_roc_curve(y_true, y_prob, path=outputs["roc_curve"]),
        plot_precision_recall_curve(y_true, y_prob, path=outputs["precision_recall_curve"]),
        plot_calibration_curve(y_true, y_prob, path=outputs["calibration_curve"], n_bins=n_bins),
        plot_probability_distribution(y_true, y_prob, path=outputs["probability_distribution"]),
    ]
    if close:
        for fig in figs:
            plt.close(fig)
    return outputs


def use_headless_backend() -> None:
    """Switch matplotlib to the non-interactive Agg backend (for CLI runs)."""
    matplotlib.use("Agg")
