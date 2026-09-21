"""Shared helpers for the relational (historical) feature builders.

Every builder in this package follows the same contract:

* input: the raw Kaggle table(s), read with an explicit column subset and
  compact dtypes (``read_raw_table``);
* output: **one row per** ``SK_ID_CURR`` with ``float32`` feature columns whose
  names carry the source prefix (``bureau__credit_count``), verified by
  :func:`finalize_customer_table`;
* a :class:`FeatureSpec` for every output column (family, description,
  aggregation), from which the feature catalog is generated and against which
  the tests check the produced columns;
* an audit dictionary of data-quality observations (never silent fixes).

Nothing here looks at ``TARGET``: the builders receive raw historical tables
only, never the application table.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskpilot import config

logger = logging.getLogger(__name__)

ID = config.ID_COL
FLOAT = "float32"


@dataclass(frozen=True)
class FeatureSpec:
    """Documentation of one engineered feature (drives the catalog and the tests)."""

    name: str
    family: str
    description: str
    aggregation: str
    count_like: bool = False  # 0 is the true value for a customer with no history rows


def read_raw_table(
    source: str,
    *,
    usecols: Sequence[str],
    dtypes: Mapping[str, str],
    raw_dir: Path = config.RAW_DATA_DIR,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read a raw table with an explicit column subset and compact dtypes."""
    path = raw_dir / config.RELATIONAL_TABLES[source]["filename"]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download it with 'python -m riskpilot.data.download --files "
            "relational'."
        )
    t0 = time.perf_counter()
    frame = pd.read_csv(path, usecols=list(usecols), dtype=dict(dtypes), nrows=nrows)
    logger.info(
        "Read %s: %s rows x %d columns, %.0f MB, %.1fs",
        path.name,
        f"{len(frame):,}",
        frame.shape[1],
        frame.memory_usage(deep=True).sum() / 1e6,
        time.perf_counter() - t0,
    )
    return frame


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """``numerator / denominator`` with NaN wherever the denominator is missing or <= 0."""
    den = denominator.where(denominator > 0)
    return (numerator / den).astype(FLOAT)


def share_of(flags: pd.Series, by: pd.Series) -> pd.Series:
    """Mean of a boolean/0-1 series per group (share of rows where the flag holds)."""
    return flags.astype(FLOAT).groupby(by).mean()


def group_agg(frame: pd.DataFrame, by: str, spec: Mapping[str, tuple[str, str]]) -> pd.DataFrame:
    """Named aggregation ``{out_name: (column, stat)}`` grouped by ``by``."""
    out = frame.groupby(by, sort=True).agg(**{k: v for k, v in spec.items()})
    return out


def finalize_customer_table(
    features: pd.DataFrame,
    *,
    prefix: str,
    specs: Sequence[FeatureSpec],
) -> pd.DataFrame:
    """Prefix names, cast to float32, verify the grain and the spec coverage.

    ``features`` must be indexed by ``SK_ID_CURR`` (one row per customer) and its
    columns must match ``specs`` exactly (order is taken from ``specs``).
    """
    expected = [s.name for s in specs]
    missing = [c for c in expected if c not in features.columns]
    extra = [c for c in features.columns if c not in expected]
    if missing or extra:
        raise ValueError(f"{prefix}: feature/spec mismatch. missing={missing} extra={extra}")
    if features.index.name != ID:
        raise ValueError(f"{prefix}: features must be indexed by {ID}, got {features.index.name!r}")
    if not features.index.is_unique:
        raise ValueError(
            f"{prefix}: customer grain violated ({features.index.duplicated().sum()} duplicate ids)"
        )
    out = features[expected].astype(FLOAT)
    out.columns = [f"{prefix}__{c}" for c in expected]
    out = out.reset_index()
    out[ID] = out[ID].astype("int32")
    return out


def specs_to_catalog(
    specs: Iterable[FeatureSpec],
    *,
    source: str,
    prefix: str,
    leakage: str,
) -> pd.DataFrame:
    """Catalog rows (one per feature) for ``relational_feature_catalog.csv``."""
    rows = [
        {
            "feature": f"{prefix}__{s.name}",
            "source": source,
            "family": s.family,
            "description": s.description,
            "aggregation": s.aggregation,
            "count_like": s.count_like,
            "leakage_assessment": leakage,
        }
        for s in specs
    ]
    return pd.DataFrame(rows)


def clip_nonnegative(series: pd.Series) -> pd.Series:
    """Clip negative values to zero (used where negatives are documented anomalies)."""
    return series.clip(lower=0)


def latest_per_group(frame: pd.DataFrame, by: str, month_col: str) -> pd.DataFrame:
    """Rows holding the most recent ``month_col`` (closest to 0) within each group."""
    idx = frame.groupby(by, sort=False)[month_col].idxmax()
    return frame.loc[idx]


def audit_series(series: pd.Series) -> dict[str, Any]:
    """Compact data-quality summary of a numeric column (JSON-friendly)."""
    s = series.dropna()
    return {
        "n": int(series.size),
        "n_missing": int(series.isna().sum()),
        "n_negative": int((s < 0).sum()) if s.size else 0,
        "n_zero": int((s == 0).sum()) if s.size else 0,
        "min": float(s.min()) if s.size else None,
        "median": float(s.median()) if s.size else None,
        "q99": float(s.quantile(0.99)) if s.size else None,
        "max": float(s.max()) if s.size else None,
    }


def value_counts_dict(series: pd.Series, top: int = 12) -> dict[str, int]:
    counts = series.value_counts(dropna=False).head(top)
    return {str(k): int(v) for k, v in counts.items()}


def to_float32_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast every float64 column to float32 in place (memory)."""
    for col in frame.columns:
        if frame[col].dtype == np.float64:
            frame[col] = frame[col].astype(FLOAT)
    return frame
