"""Leakage-safe preprocessing for the Logistic Regression baseline.

Design decisions (see ``reports/baseline_report.md`` for the reasoning):

* Everything that *learns* from data (medians, scaling statistics, one-hot
  vocabularies) lives inside a scikit-learn ``Pipeline`` / ``ColumnTransformer``
  and is therefore fitted exclusively on the training split.
* Sentinel values (``DAYS_EMPLOYED == 365243``) are mapped to ``NaN`` by a
  stateless transformer *before* imputation, so the median imputer and the
  missing-value indicator treat them as "unknown" instead of as a huge
  magnitude that would dominate a linear model.
* Numeric branch: median imputation with missing indicators, then
  standardization (Logistic Regression with L2 regularization needs features on
  a comparable scale for the penalty to be meaningful and for lbfgs to converge).
* Categorical branch: explicit "missing" category, then one-hot encoding with
  ``handle_unknown="ignore"`` so unseen categories at inference time become an
  all-zero row instead of an error. Cardinality in this table is modest
  (max ~60 levels), so one-hot is memory-safe; the encoder emits a sparse block.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from riskpilot import config
from riskpilot.data.validation import FeatureTypes, split_feature_types

__all__ = [
    "FeatureTypes",
    "build_baseline_pipeline",
    "build_preprocessor",
    "clean_features",
    "infer_feature_types",
    "make_sentinel_transformer",
    "replace_sentinels",
]


def infer_feature_types(X: pd.DataFrame) -> FeatureTypes:
    """Infer numeric / categorical feature groups from a *training* feature frame.

    ``X`` must already exclude the target and identifier columns. Inference
    should be run on the training split so that the feature list is part of what
    is "fitted" on training data only.
    """
    return split_feature_types(X, exclude=())


def replace_sentinels(
    X: pd.DataFrame,
    sentinels: Mapping[str, float] = config.SENTINEL_VALUES,
) -> pd.DataFrame:
    """Return a copy of ``X`` where configured sentinel values are replaced by NaN.

    Stateless (nothing is learned), so it cannot leak information between splits.
    Columns absent from ``X`` are ignored.
    """
    out = X.copy()
    for col, sentinel in sentinels.items():
        if col in out.columns:
            out[col] = out[col].mask(out[col] == sentinel)
    return out


def clean_features(
    X: pd.DataFrame,
    sentinels: Mapping[str, float] = config.SENTINEL_VALUES,
) -> pd.DataFrame:
    """Stateless input canonicalization applied before any fitted step.

    * sentinel values become NaN (:func:`replace_sentinels`);
    * ``None`` in string columns becomes ``np.nan`` so that scikit-learn's
      imputer (which only recognizes ``np.nan`` in object arrays) sees it as
      missing. ``pandas.read_csv`` already yields NaN, but in-memory frames may
      carry ``None``.
    """
    out = replace_sentinels(X, sentinels)
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    if len(object_cols):
        out[object_cols] = out[object_cols].where(out[object_cols].notna(), np.nan)
    return out


def make_sentinel_transformer(
    sentinels: Mapping[str, float] = config.SENTINEL_VALUES,
) -> FunctionTransformer:
    """Wrap :func:`clean_features` as a scikit-learn transformer."""
    return FunctionTransformer(
        clean_features,
        kw_args={"sentinels": dict(sentinels)},
        feature_names_out="one-to-one",
        validate=False,
    )


def build_preprocessor(
    feature_types: FeatureTypes,
    *,
    min_frequency: int | float | None = None,
) -> ColumnTransformer:
    """Build the numeric + categorical ``ColumnTransformer``.

    Parameters
    ----------
    feature_types:
        Column groups inferred from the training split.
    min_frequency:
        Optional ``OneHotEncoder(min_frequency=...)`` to fold rare categories into
        an "infrequent" bucket. ``None`` keeps every level seen in training.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    transformers = []
    if feature_types.numeric:
        transformers.append(("num", numeric_pipeline, list(feature_types.numeric)))
    if feature_types.categorical:
        transformers.append(("cat", categorical_pipeline, list(feature_types.categorical)))
    if not transformers:
        raise ValueError("No features to preprocess: both numeric and categorical lists are empty.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_baseline_pipeline(
    feature_types: FeatureTypes,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str | dict | None = None,
    sentinels: Mapping[str, float] = config.SENTINEL_VALUES,
    min_frequency: int | float | None = None,
    random_state: int = config.RANDOM_STATE,
) -> Pipeline:
    """End-to-end ``raw features -> preprocessing -> LogisticRegression`` pipeline.

    ``class_weight`` defaults to ``None`` on purpose: the objective is to estimate
    ``P(Y = 1 | X)``, and re-weighting the classes would shift predicted
    probabilities away from the true base rate. Ranking metrics (ROC-AUC, PR-AUC)
    are largely unaffected by class weights for Logistic Regression, while
    probabilistic metrics (log loss, Brier) would degrade.
    """
    model = LogisticRegression(
        C=C,  # L2 penalty is scikit-learn's default (l1_ratio=0)
        solver="lbfgs",
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("clean", make_sentinel_transformer(sentinels)),
            ("preprocess", build_preprocessor(feature_types, min_frequency=min_frequency)),
            ("model", model),
        ]
    )
