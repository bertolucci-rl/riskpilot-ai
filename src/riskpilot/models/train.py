"""Train and evaluate the Logistic Regression baseline end to end.

Run from the command line::

    python -m riskpilot.models.train
    python -m riskpilot.models.train --nrows 20000   # quick smoke run

Steps: load -> validate target -> stratified split -> fit pipeline on the
training split only -> evaluate probabilities on the untouched test split ->
save metrics (JSON), figures (PNG), split membership (CSV) and the fitted
pipeline (joblib).
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from riskpilot import __version__, config
from riskpilot.data.load import load_application_train, split_features_target
from riskpilot.data.validation import validate_target
from riskpilot.features.preprocessing import build_baseline_pipeline, infer_feature_types
from riskpilot.models.evaluate import (
    calibration_table,
    compute_metrics,
    make_evaluation_figures,
    save_metrics,
    use_headless_backend,
)

logger = logging.getLogger(__name__)


def _display_path(path: str | Path) -> str:
    """Portable string for a path: relative to the project root when inside it.

    Keeps user-specific absolute paths out of the committed metrics file and out
    of notebook outputs; paths outside the project stay absolute.
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


@dataclass
class BaselineConfig:
    """All knobs of a baseline run. Paths default to the project layout."""

    data_path: Path = config.APPLICATION_TRAIN_PATH
    nrows: int | None = None
    test_size: float = config.TEST_SIZE
    random_state: int = config.RANDOM_STATE
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = None
    n_calibration_bins: int = 10
    prefix: str = "baseline"
    figures_dir: Path = config.FIGURES_DIR
    metrics_dir: Path = config.METRICS_DIR
    models_dir: Path = config.MODELS_DIR
    processed_dir: Path = config.PROCESSED_DATA_DIR
    save_model: bool = True
    save_split: bool = True
    sentinels: dict[str, float] = field(default_factory=lambda: dict(config.SENTINEL_VALUES))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = _display_path(value)
        return out


@dataclass
class BaselineResult:
    pipeline: Pipeline
    payload: dict[str, Any]
    y_test: pd.Series
    y_prob_test: np.ndarray
    metrics_path: Path
    model_path: Path | None


def make_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified random holdout split.

    The application table carries no application date, so a chronological split
    is not possible; stratifying on the target keeps the default rate identical
    in both partitions, which matters for a ~8% prevalence problem.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def _fit_with_convergence_check(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> bool:
    """Fit and report (not suppress) whether a ConvergenceWarning was raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X, y)
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if warned:
        logger.warning(
            "LogisticRegression did not converge within max_iter=%d; results are recorded "
            "but the run should be investigated (scaling, C, max_iter).",
            pipeline["model"].max_iter,
        )
    return warned


def _penalty_name(model: LogisticRegression) -> str:
    """Name of the penalty actually applied, independent of the scikit-learn version.

    scikit-learn 1.8 deprecated ``LogisticRegression(penalty=...)`` in favour of
    ``l1_ratio`` (0 = L2, 1 = L1, in between = elastic net) and ``C=inf`` for no
    penalty; the ``penalty`` attribute then holds the placeholder ``"deprecated"``.
    """
    penalty = getattr(model, "penalty", "deprecated")
    if penalty != "deprecated":
        return "none" if penalty is None else str(penalty)
    if np.isinf(model.C):
        return "none"
    l1_ratio = model.l1_ratio or 0.0
    if l1_ratio == 0.0:
        return "l2"
    if l1_ratio == 1.0:
        return "l1"
    return "elasticnet"


def run_baseline(cfg: BaselineConfig | None = None) -> BaselineResult:
    """Execute the full baseline experiment and persist its artifacts."""
    cfg = cfg or BaselineConfig()
    started = time.perf_counter()

    df = load_application_train(cfg.data_path, nrows=cfg.nrows)
    validate_target(df[config.TARGET_COL])
    X, y = split_features_target(df)
    ids = df[config.ID_COL] if config.ID_COL in df.columns else pd.Series(df.index, name="row")
    del df

    X_train, X_test, y_train, y_test = make_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    logger.info(
        "Split: train=%d (prevalence %.4f) | test=%d (prevalence %.4f)",
        len(y_train),
        y_train.mean(),
        len(y_test),
        y_test.mean(),
    )

    # Feature typing is inferred from the training split only.
    feature_types = infer_feature_types(X_train)
    pipeline = build_baseline_pipeline(
        feature_types,
        C=cfg.C,
        max_iter=cfg.max_iter,
        class_weight=cfg.class_weight,
        sentinels=cfg.sentinels,
        random_state=cfg.random_state,
    )

    fit_start = time.perf_counter()
    convergence_warning = _fit_with_convergence_check(pipeline, X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    model = pipeline["model"]
    n_iter = int(np.asarray(model.n_iter_).max())
    n_transformed = int(pipeline["preprocess"].get_feature_names_out().shape[0])
    logger.info(
        "Fitted in %.1fs | lbfgs iterations=%d | transformed features=%d | converged=%s",
        fit_seconds,
        n_iter,
        n_transformed,
        not convergence_warning,
    )

    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    metrics_test = compute_metrics(y_test, y_prob_test, n_bins=cfg.n_calibration_bins)
    metrics_train = compute_metrics(y_train, y_prob_train, n_bins=cfg.n_calibration_bins)
    table = calibration_table(y_test, y_prob_test, n_bins=cfg.n_calibration_bins)

    figures = make_evaluation_figures(
        y_test,
        y_prob_test,
        figures_dir=cfg.figures_dir,
        prefix=cfg.prefix,
        n_bins=cfg.n_calibration_bins,
    )

    model_path: Path | None = None
    if cfg.save_model:
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = cfg.models_dir / f"{cfg.prefix}_logistic_regression.joblib"
        joblib.dump(pipeline, model_path)
        logger.info("Saved fitted pipeline to %s", _display_path(model_path))

    split_path: Path | None = None
    if cfg.save_split:
        cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        split_path = cfg.processed_dir / f"{cfg.prefix}_split_membership.csv"
        membership = pd.DataFrame({ids.name or "row": ids})
        membership["split"] = "train"
        membership.loc[X_test.index, "split"] = "test"
        membership.to_csv(split_path, index=False)

    payload: dict[str, Any] = {
        "run": {
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "riskpilot_version": __version__,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "total_seconds": round(time.perf_counter() - started, 1),
        },
        "config": cfg.to_dict(),
        "data": {
            "path": _display_path(cfg.data_path),
            "n_rows": int(len(y)),
            "n_raw_features": int(X.shape[1]),
            "n_numeric_features": len(feature_types.numeric),
            "n_categorical_features": len(feature_types.categorical),
            "n_transformed_features": n_transformed,
        },
        "split": {
            "strategy": "stratified_random_holdout",
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "prevalence_train": float(y_train.mean()),
            "prevalence_test": float(y_test.mean()),
            "membership_file": _display_path(split_path) if split_path else None,
        },
        "model": {
            "estimator": "LogisticRegression",
            "solver": model.solver,
            "penalty": _penalty_name(model),
            "C": model.C,
            "class_weight": model.class_weight,
            "max_iter": model.max_iter,
            "n_iter": n_iter,
            "converged": not convergence_warning,
            "fit_seconds": round(fit_seconds, 1),
            "intercept": float(model.intercept_[0]),
            "n_coefficients": int(model.coef_.shape[1]),
        },
        "metrics": {"test": metrics_test, "train": metrics_train},
        "calibration_table_test": table.to_dict(orient="records"),
        "artifacts": {
            "figures": {k: _display_path(v) for k, v in figures.items()},
            "model": _display_path(model_path) if model_path else None,
        },
    }
    metrics_path = save_metrics(payload, cfg.metrics_dir / f"{cfg.prefix}_metrics.json")
    logger.info("Saved metrics to %s", _display_path(metrics_path))

    return BaselineResult(
        pipeline=pipeline,
        payload=payload,
        y_test=y_test,
        y_prob_test=y_prob_test,
        metrics_path=metrics_path,
        model_path=model_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riskpilot.models.train",
        description="Train and evaluate the RiskPilot Logistic Regression baseline.",
    )
    parser.add_argument("--data-path", type=Path, default=config.APPLICATION_TRAIN_PATH)
    parser.add_argument("--nrows", type=int, default=None, help="Row limit for smoke runs.")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularization strength.")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--prefix", default="baseline", help="Artifact filename prefix.")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--no-save-split", action="store_true")
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
    cfg = BaselineConfig(
        data_path=args.data_path,
        nrows=args.nrows,
        test_size=args.test_size,
        random_state=args.random_state,
        C=args.C,
        max_iter=args.max_iter,
        prefix=args.prefix,
        figures_dir=args.figures_dir,
        metrics_dir=args.metrics_dir,
        models_dir=args.models_dir,
        processed_dir=args.processed_dir,
        save_model=not args.no_save_model,
        save_split=not args.no_save_split,
    )
    try:
        result = run_baseline(cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    m = result.payload["metrics"]["test"]
    print(
        "\nBaseline (test set)\n"
        f"  ROC-AUC            : {m['roc_auc']:.4f}\n"
        f"  Average precision  : {m['average_precision']:.4f}  "
        f"(prevalence {m['prevalence']:.4f})\n"
        f"  Log loss           : {m['log_loss']:.4f}  "
        f"(prevalence baseline {m['log_loss_prevalence_baseline']:.4f})\n"
        f"  Brier score        : {m['brier_score']:.4f}  "
        f"(prevalence baseline {m['brier_prevalence_baseline']:.4f}, "
        f"skill {m['brier_skill_score']:.3f})\n"
        f"  ECE (10 q-bins)    : {m['expected_calibration_error']:.4f}\n"
        f"  Converged          : {result.payload['model']['converged']} "
        f"({result.payload['model']['n_iter']} iterations)\n"
        f"  Metrics file       : {result.metrics_path}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
