"""Dataset sanity checks and data-quality observations.

These functions *observe* and *report*; they never drop or alter columns. The
decision about what to do with a suspicious column is made explicitly in the
preprocessing code and documented in the report, not hidden in a validator.

All return values are plain ``dict`` / ``DataFrame`` objects so they can be
serialized to JSON for the notebooks, tests and the technical report.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from riskpilot import config

logger = logging.getLogger(__name__)

# Tokens that the Home Credit tables use to encode "unknown" inside string columns.
UNKNOWN_TOKENS: tuple[str, ...] = ("XNA", "XAP", "Unknown", "unknown", "N/A", "NA", "")

# Column-name fragments that would suggest post-outcome information (leakage).
LEAKAGE_NAME_PATTERNS: tuple[str, ...] = (
    "TARGET",
    "DEFAULT",
    "DPD",
    "OVERDUE",
    "LATE",
    "DELINQ",
    "WRITEOFF",
    "CHARGE_OFF",
    "COLLECT",
)


@dataclass(frozen=True)
class FeatureTypes:
    """Column names grouped by the preprocessing branch they belong to."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return [*self.numeric, *self.categorical]


def _py(value: Any) -> Any:
    """Convert numpy scalars to plain Python so dicts serialize to JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
def validate_target(
    y: pd.Series,
    *,
    allowed_values: tuple[int, ...] = config.ALLOWED_TARGET_VALUES,
) -> None:
    """Raise ``ValueError`` unless ``y`` is a complete binary target with both classes."""
    if y.isna().any():
        raise ValueError(f"Target contains {int(y.isna().sum())} missing value(s).")
    observed = set(pd.unique(y))
    unexpected = observed - set(allowed_values)
    if unexpected:
        raise ValueError(f"Target contains unexpected value(s): {sorted(map(_py, unexpected))}.")
    if len(observed) < 2:
        raise ValueError(f"Target has a single class {sorted(map(_py, observed))}; cannot model.")


def summarize_target(y: pd.Series) -> dict[str, Any]:
    """Counts, prevalence and missingness of the binary target."""
    counts = y.value_counts(dropna=False)
    n = int(len(y))
    n_missing = int(y.isna().sum())
    positives = int((y == 1).sum())
    return {
        "n": n,
        "n_missing": n_missing,
        "value_counts": {str(_py(k)): int(v) for k, v in counts.items()},
        "n_positive": positives,
        "prevalence": positives / n if n else float("nan"),
        "imbalance_ratio": (n - n_missing - positives) / positives if positives else float("inf"),
    }


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def dataset_overview(df: pd.DataFrame, *, id_col: str = config.ID_COL) -> dict[str, Any]:
    """Shape, memory, duplicate rows / ids and dtype composition."""
    dtype_counts = df.dtypes.astype(str).value_counts()
    overview: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "dtype_counts": {str(k): int(v) for k, v in dtype_counts.items()},
    }
    if id_col in df.columns:
        overview["id_column"] = id_col
        overview["n_duplicate_ids"] = int(df[id_col].duplicated().sum())
        overview["id_is_unique"] = bool(df[id_col].is_unique)
    return overview


def split_feature_types(
    df: pd.DataFrame,
    *,
    exclude: tuple[str, ...] = (config.ID_COL, config.TARGET_COL),
) -> FeatureTypes:
    """Group feature columns into numeric and categorical branches.

    Boolean and string-like columns are treated as categorical. Any column that
    fits neither branch (e.g. datetimes) raises so that it is handled explicitly.
    """
    features = df.drop(columns=[c for c in exclude if c in df.columns])
    categorical = features.select_dtypes(include=["object", "string", "category", "bool"])
    numeric = features.select_dtypes(include=["number"])
    leftover = set(features.columns) - set(categorical.columns) - set(numeric.columns)
    if leftover:
        raise TypeError(f"Columns with unsupported dtypes for preprocessing: {sorted(leftover)}")
    return FeatureTypes(numeric=list(numeric.columns), categorical=list(categorical.columns))


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and fractions, most-missing first."""
    n_missing = df.isna().sum()
    n_rows = len(df)
    table = pd.DataFrame(
        {
            "column": n_missing.index,
            "dtype": df.dtypes.astype(str).values,
            "n_missing": n_missing.values,
            "frac_missing": (n_missing / n_rows).values if n_rows else 0.0,
        }
    )
    return table.sort_values("frac_missing", ascending=False, kind="stable").reset_index(drop=True)


def missingness_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate view of the missingness table."""
    table = missingness_table(df)
    return {
        "n_columns_with_missing": int((table["n_missing"] > 0).sum()),
        "n_columns_over_50pct_missing": int((table["frac_missing"] > 0.5).sum()),
        "n_columns_over_20pct_missing": int((table["frac_missing"] > 0.2).sum()),
        "total_missing_fraction": float(table["n_missing"].sum() / max(df.size, 1)),
        "top_missing": {
            row.column: float(row.frac_missing) for row in table.head(10).itertuples()
        },
    }


# --------------------------------------------------------------------------- #
# Feature quality
# --------------------------------------------------------------------------- #
def constant_features(
    df: pd.DataFrame,
    *,
    near_constant_threshold: float = 0.99,
    exclude: tuple[str, ...] = (config.ID_COL, config.TARGET_COL),
) -> pd.DataFrame:
    """Columns whose most frequent value covers at least the threshold share of rows.

    ``is_constant`` marks columns with a single distinct non-null value.
    """
    records = []
    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        counts = series.value_counts(dropna=True)
        n_unique = int(len(counts))
        top_frac = float(counts.iloc[0] / len(series)) if n_unique and len(series) else 0.0
        if n_unique <= 1 or top_frac >= near_constant_threshold:
            records.append(
                {
                    "column": col,
                    "n_unique": n_unique,
                    "top_value": _py(counts.index[0]) if n_unique else None,
                    "top_frac": top_frac,
                    "is_constant": n_unique <= 1,
                }
            )
    columns = ["column", "n_unique", "top_value", "top_frac", "is_constant"]
    table = pd.DataFrame.from_records(records, columns=columns)
    return table.sort_values("top_frac", ascending=False, kind="stable").reset_index(drop=True)


def categorical_cardinality(
    df: pd.DataFrame,
    *,
    exclude: tuple[str, ...] = (config.ID_COL, config.TARGET_COL),
) -> pd.DataFrame:
    """Distinct-value counts for every categorical column (sizes the one-hot encoding)."""
    types = split_feature_types(df, exclude=exclude)
    records = [
        {
            "column": col,
            "n_unique": int(df[col].nunique(dropna=True)),
            "frac_missing": float(df[col].isna().mean()),
        }
        for col in types.categorical
    ]
    table = pd.DataFrame.from_records(records, columns=["column", "n_unique", "frac_missing"])
    return table.sort_values("n_unique", ascending=False, kind="stable").reset_index(drop=True)


def high_cardinality_features(
    df: pd.DataFrame,
    *,
    threshold: int = 20,
    exclude: tuple[str, ...] = (config.ID_COL, config.TARGET_COL),
) -> pd.DataFrame:
    """Categorical columns with more than ``threshold`` distinct values."""
    table = categorical_cardinality(df, exclude=exclude)
    return table[table["n_unique"] > threshold].reset_index(drop=True)


def suspicious_values_report(
    df: pd.DataFrame,
    *,
    sentinels: dict[str, float] = config.SENTINEL_VALUES,
    extreme_ratio: float = 100.0,
) -> dict[str, Any]:
    """Flag impossible or suspicious values without altering anything.

    Checks performed:

    * ``unknown_tokens``: string columns containing XNA / Unknown style tokens.
    * ``days_columns_positive``: ``DAYS_*`` columns are documented as days
      *before* the application (<= 0); positive values are anomalies.
    * ``sentinels``: how often each configured sentinel value occurs.
    * ``extreme_numeric``: numeric columns whose max exceeds ``extreme_ratio``
      times the 99th percentile (heavy tails or data-entry outliers).
    * ``negative_amounts``: ``AMT_*`` / ``CNT_*`` columns with negative values.
    """
    report: dict[str, Any] = {
        "unknown_tokens": {},
        "days_columns_positive": {},
        "sentinels": {},
        "extreme_numeric": {},
        "negative_amounts": {},
    }
    types = split_feature_types(df)
    n_rows = len(df)

    for col in types.categorical:
        counts = df[col].value_counts(dropna=True)
        found = {str(k): int(v) for k, v in counts.items() if str(k) in UNKNOWN_TOKENS}
        if found:
            report["unknown_tokens"][col] = found

    for col in types.numeric:
        series = df[col]
        if col.startswith("DAYS_"):
            n_pos = int((series > 0).sum())
            if n_pos:
                report["days_columns_positive"][col] = {
                    "n_positive": n_pos,
                    "frac_positive": float(n_pos / n_rows) if n_rows else 0.0,
                    "max": float(series.max()),
                }
        if col in sentinels:
            n_sentinel = int((series == sentinels[col]).sum())
            report["sentinels"][col] = {
                "sentinel": float(sentinels[col]),
                "n": n_sentinel,
                "frac": float(n_sentinel / n_rows) if n_rows else 0.0,
            }
        non_null = series.dropna()
        if len(non_null) > 10:
            q99 = float(non_null.quantile(0.99))
            max_value = float(non_null.max())
            if q99 > 0 and max_value > extreme_ratio * q99:
                report["extreme_numeric"][col] = {
                    "max": max_value,
                    "q99": q99,
                    "median": float(non_null.median()),
                    "ratio_max_to_q99": max_value / q99,
                }
        if col.startswith(("AMT_", "CNT_")):
            n_neg = int((series < 0).sum())
            if n_neg:
                report["negative_amounts"][col] = n_neg
    return report


# --------------------------------------------------------------------------- #
# Leakage screening
# --------------------------------------------------------------------------- #
def leakage_screen(
    df: pd.DataFrame,
    *,
    target_col: str = config.TARGET_COL,
    auc_threshold: float = 0.80,
    name_patterns: tuple[str, ...] = LEAKAGE_NAME_PATTERNS,
) -> dict[str, Any]:
    """Screen features for target leakage.

    Two complementary heuristics are used:

    * name-based: columns whose name suggests post-outcome information;
    * univariate separation: for every numeric feature, ``max(AUC, 1 - AUC)`` of
      the raw values against the target on non-missing rows. Legitimate
      application-time features rarely exceed roughly 0.7 on their own, so
      anything above ``auc_threshold`` is flagged for manual inspection.

    This is a *screen*, not a verdict. Flagged columns are reported, not removed.
    """
    y = df[target_col]
    types = split_feature_types(df, exclude=(config.ID_COL, target_col))

    name_flags = [
        col
        for col in df.columns
        if col != target_col and any(re.search(p, col, flags=re.IGNORECASE) for p in name_patterns)
    ]

    separation: dict[str, float] = {}
    for col in types.numeric:
        mask = df[col].notna()
        if mask.sum() < 50 or y[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], df.loc[mask, col])
        separation[col] = float(max(auc, 1.0 - auc))

    ranked = sorted(separation.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "name_based_flags": name_flags,
        "auc_threshold": auc_threshold,
        "high_separation_features": {c: v for c, v in ranked if v >= auc_threshold},
        "top_univariate_auc": dict(ranked[:15]),
    }


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def run_validation(df: pd.DataFrame, *, target_col: str = config.TARGET_COL) -> dict[str, Any]:
    """Run every check and return a JSON-serializable report."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is absent from the DataFrame.")
    validate_target(df[target_col])

    types = split_feature_types(df)
    constants = constant_features(df)
    high_card = high_cardinality_features(df)

    report = {
        "overview": dataset_overview(df),
        "target": summarize_target(df[target_col]),
        "feature_types": {
            "n_numeric": len(types.numeric),
            "n_categorical": len(types.categorical),
            "categorical": types.categorical,
        },
        "missingness": missingness_summary(df),
        "constant_features": constants.to_dict(orient="records"),
        "high_cardinality_features": high_card.to_dict(orient="records"),
        "categorical_cardinality": categorical_cardinality(df).to_dict(orient="records"),
        "suspicious_values": suspicious_values_report(df),
        "leakage": leakage_screen(df, target_col=target_col),
    }
    logger.info(
        "Validation: %d rows, %d cols, prevalence=%.4f, %d constant/near-constant, "
        "%d high-cardinality categoricals",
        report["overview"]["n_rows"],
        report["overview"]["n_columns"],
        report["target"]["prevalence"],
        len(report["constant_features"]),
        len(report["high_cardinality_features"]),
    )
    return report
