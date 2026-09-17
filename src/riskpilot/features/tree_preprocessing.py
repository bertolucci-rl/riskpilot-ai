"""Leakage-safe preprocessing for gradient-boosting (tree) models.

Trees do not need what the linear baseline needs. The Logistic Regression
pipeline (``riskpilot.features.preprocessing``) imputes, adds missing
indicators, standardizes and one-hot encodes because a linear model with an L2
penalty requires comparable scales and cannot consume missing values or strings.
Gradient-boosted trees are invariant to monotone rescaling, learn a default
direction for missing values at every split, and both LightGBM and XGBoost
split natively on categorical columns. Forcing the linear preprocessing on them
would only add 192 redundant columns (62 indicators + 146 one-hot) and a dense
0.6 GB matrix.

:class:`TreePreprocessor` therefore does the minimum that is still learned on
the training split only:

* stateless cleaning (sentinel ``DAYS_EMPLOYED == 365243`` -> NaN, ``None`` ->
  NaN), shared with the linear pipeline;
* numeric columns are cast to ``float32`` and otherwise left alone: no
  imputation, no scaling;
* categorical columns become ``pandas.Categorical`` with the vocabulary seen
  in the training split, so category *codes* are identical at fit and predict
  time and unseen levels become missing. Both libraries consume this dtype
  directly, each with its own split algorithm (documented in the report);
* two optional, explicitly labelled treatments used by the ablations:
  ``heavy_tail`` (``log1p`` or winsorization of skewed non-negative columns,
  bounds learned on the training split) and ``missing_indicators`` (``"none"``,
  ``"all"`` or ``"deduplicated"``).

The transformer refuses to fit when the target or the identifier column is
present, so the target can never be used as a feature by accident.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from riskpilot import config
from riskpilot.features.preprocessing import clean_features, infer_feature_types

__all__ = [
    "HeavyTail",
    "MissingIndicators",
    "TreePreprocessor",
    "duplicate_missing_indicator_groups",
    "select_heavy_tail_columns",
]

HeavyTail = Literal["none", "log1p", "winsorize"]
MissingIndicators = Literal["none", "all", "deduplicated"]

FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (config.TARGET_COL, config.ID_COL)


def select_heavy_tail_columns(
    X: pd.DataFrame,
    *,
    min_unique: int = 20,
    min_skew: float = 1.0,
) -> list[str]:
    """Data-driven list of skewed, non-negative, unbounded numeric columns.

    A column qualifies when, on the frame it is given (the training split), it
    is numeric, has no negative value, is not a normalized score bounded in
    ``[0, 1]`` (``max > 1``), has at least ``min_unique`` distinct values (which
    excludes 0/1 flags and small integer codes) and has skewness above
    ``min_skew``. On ``application_train`` this selects the four monetary
    ``AMT_*`` amounts, the social-circle observation counts, the credit-bureau
    enquiry counts and ``OWN_CAR_AGE``; the bounded building-description
    statistics are excluded on purpose.
    """
    selected: list[str] = []
    for col in X.select_dtypes(include=["number"]).columns:
        s = X[col].dropna()
        if s.empty or s.nunique() < min_unique:
            continue
        if float(s.min()) < 0.0 or float(s.max()) <= 1.0:
            continue
        if float(s.skew()) > min_skew:
            selected.append(str(col))
    return selected


def duplicate_missing_indicator_groups(X: pd.DataFrame, columns: Sequence[str]) -> list[list[str]]:
    """Group columns whose missingness patterns are identical on ``X``.

    Returns one list per group, in first-seen column order; singletons are
    groups of size one. Used to verify, not assume, that the redundant missing
    indicators found in Milestone 1 are exact duplicates before de-duplicating.
    """
    groups: dict[bytes, list[str]] = {}
    for col in columns:
        key = np.packbits(X[col].isna().to_numpy()).tobytes()
        groups.setdefault(key, []).append(str(col))
    return list(groups.values())


class TreePreprocessor(BaseEstimator, TransformerMixin):
    """``raw features -> DataFrame(float32 numerics, pandas categoricals)``.

    Parameters
    ----------
    sentinels:
        Sentinel values mapped to NaN before anything else (stateless).
    heavy_tail:
        ``"none"`` (default), ``"log1p"`` or ``"winsorize"``. Applied only to
        ``heavy_tail_columns`` (explicit list) or, when that is ``None``, to the
        columns returned by :func:`select_heavy_tail_columns` on the training
        split. Winsorization bounds are training-split quantiles.
    winsor_quantiles:
        Lower / upper quantiles for ``"winsorize"``.
    missing_indicators:
        ``"none"`` (default: the libraries handle NaN natively), ``"all"`` (one
        0/1 column per numeric column with a missing value in training, as the
        linear baseline does) or ``"deduplicated"`` (one column per group of
        identical missingness patterns, verified on the training split).
    float_dtype:
        dtype of the numeric block. ``float32`` halves memory and is what both
        libraries use internally.
    """

    def __init__(
        self,
        *,
        sentinels: Mapping[str, float] | None = None,
        heavy_tail: HeavyTail = "none",
        heavy_tail_columns: Sequence[str] | None = None,
        winsor_quantiles: tuple[float, float] = (0.001, 0.999),
        missing_indicators: MissingIndicators = "none",
        float_dtype: str = "float32",
    ) -> None:
        self.sentinels = sentinels
        self.heavy_tail = heavy_tail
        self.heavy_tail_columns = heavy_tail_columns
        self.winsor_quantiles = winsor_quantiles
        self.missing_indicators = missing_indicators
        self.float_dtype = float_dtype

    # ------------------------------------------------------------------ #
    def _sentinels(self) -> dict[str, float]:
        return dict(config.SENTINEL_VALUES if self.sentinels is None else self.sentinels)

    def _validate_params(self) -> None:
        if self.heavy_tail not in ("none", "log1p", "winsorize"):
            raise ValueError(
                f"heavy_tail must be 'none', 'log1p' or 'winsorize', got {self.heavy_tail!r}."
            )
        if self.missing_indicators not in ("none", "all", "deduplicated"):
            raise ValueError(
                "missing_indicators must be 'none', 'all' or 'deduplicated', "
                f"got {self.missing_indicators!r}."
            )
        lo, hi = self.winsor_quantiles
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(
                f"winsor_quantiles must satisfy 0 <= lo < hi <= 1, got {self.winsor_quantiles}."
            )

    @staticmethod
    def _check_no_target(X: pd.DataFrame) -> None:
        present = [c for c in FORBIDDEN_FEATURE_COLUMNS if c in X.columns]
        if present:
            raise ValueError(
                f"Columns {present} must not be part of the feature frame "
                "(target / identifier leakage). Use split_features_target first."
            )

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: Any = None) -> TreePreprocessor:
        self._validate_params()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TreePreprocessor expects a pandas DataFrame.")
        self._check_no_target(X)

        Xc = clean_features(X, self._sentinels())
        types = infer_feature_types(Xc)
        if not types.all:
            raise ValueError("No features to preprocess.")
        self.numeric_columns_ = list(types.numeric)
        self.categorical_columns_ = list(types.categorical)

        # Vocabulary per categorical column, learned on the training split only.
        self.categories_: dict[str, list[Any]] = {
            col: sorted(Xc[col].dropna().unique().tolist(), key=str)
            for col in self.categorical_columns_
        }

        # Heavy-tail treatment: which columns, and (for winsorization) which bounds.
        if self.heavy_tail == "none":
            self.heavy_tail_columns_: list[str] = []
        elif self.heavy_tail_columns is None:
            self.heavy_tail_columns_ = select_heavy_tail_columns(Xc[self.numeric_columns_])
        else:
            unknown = [c for c in self.heavy_tail_columns if c not in self.numeric_columns_]
            if unknown:
                raise ValueError(f"heavy_tail_columns not numeric or not present: {unknown}")
            self.heavy_tail_columns_ = list(self.heavy_tail_columns)
        self.clip_bounds_: dict[str, tuple[float, float]] = {}
        if self.heavy_tail == "winsorize":
            lo, hi = self.winsor_quantiles
            for col in self.heavy_tail_columns_:
                s = Xc[col].dropna()
                if s.empty:
                    continue
                self.clip_bounds_[col] = (float(s.quantile(lo)), float(s.quantile(hi)))

        # Missing indicators (optional), verified on the training split.
        with_missing = [c for c in self.numeric_columns_ if Xc[c].isna().any()]
        if self.missing_indicators == "none":
            self.indicator_columns_: list[str] = []
            self.indicator_groups_: list[list[str]] = []
        elif self.missing_indicators == "all":
            self.indicator_columns_ = with_missing
            self.indicator_groups_ = [[c] for c in with_missing]
        else:
            self.indicator_groups_ = duplicate_missing_indicator_groups(Xc, with_missing)
            self.indicator_columns_ = [group[0] for group in self.indicator_groups_]

        self.feature_names_out_ = np.asarray(
            [
                *self.numeric_columns_,
                *self.categorical_columns_,
                *(f"missing_{c}" for c in self.indicator_columns_),
            ],
            dtype=object,
        )
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TreePreprocessor expects a pandas DataFrame.")
        self._check_no_target(X)
        missing_cols = [
            c for c in (*self.numeric_columns_, *self.categorical_columns_) if c not in X.columns
        ]
        if missing_cols:
            raise ValueError(f"Input is missing feature columns: {missing_cols[:10]}")

        Xc = clean_features(
            X[[*self.numeric_columns_, *self.categorical_columns_]], self._sentinels()
        )
        blocks: list[pd.DataFrame] = []

        if self.numeric_columns_:
            num = Xc[self.numeric_columns_].astype("float64")
            for col, (lo, hi) in self.clip_bounds_.items():
                num[col] = num[col].clip(lower=lo, upper=hi)
            if self.heavy_tail == "log1p":
                for col in self.heavy_tail_columns_:
                    num[col] = np.log1p(num[col].clip(lower=0.0))
            blocks.append(num.astype(self.float_dtype))

        if self.categorical_columns_:
            cat = pd.DataFrame(
                {
                    col: pd.Categorical(Xc[col], categories=self.categories_[col])
                    for col in self.categorical_columns_
                },
                index=Xc.index,
            )
            blocks.append(cat)

        if self.indicator_columns_:
            ind = pd.DataFrame(
                {
                    f"missing_{c}": Xc[c].isna().to_numpy(dtype=self.float_dtype)
                    for c in self.indicator_columns_
                },
                index=Xc.index,
            )
            blocks.append(ind)

        out = pd.concat(blocks, axis=1)
        return out[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def describe(self) -> dict[str, Any]:
        """JSON-friendly summary of what was learned (for the metrics files)."""
        check_is_fitted(self, "feature_names_out_")
        return {
            "n_numeric": len(self.numeric_columns_),
            "n_categorical": len(self.categorical_columns_),
            "n_indicators": len(self.indicator_columns_),
            "n_features_out": int(self.feature_names_out_.shape[0]),
            "categorical_cardinality": {c: len(v) for c, v in self.categories_.items()},
            "heavy_tail": self.heavy_tail,
            "heavy_tail_columns": list(self.heavy_tail_columns_),
            "clip_bounds": {c: list(b) for c, b in self.clip_bounds_.items()},
            "missing_indicators": self.missing_indicators,
            "indicator_groups": [g for g in self.indicator_groups_ if len(g) > 1],
            "float_dtype": self.float_dtype,
        }
