"""Assemble application features + customer-level relational tables, safely.

``build_feature_matrix(X, ids, sources=[...])`` left-joins the cached
customer-level table of every requested source onto the application rows.
Every join is validated ``one_to_one`` (application ids are unique, every
relational table is one row per ``SK_ID_CURR``), the row count and the index
are asserted unchanged, and a ``has_<source>`` indicator distinguishes
"no history in this source" from "history exists but the statistic is
undefined": count-like features are filled with 0 for customers without
history (0 credits is a true value), every other feature stays missing.

The identifier itself never becomes a feature; ``TARGET`` is refused.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from riskpilot import config
from riskpilot.features.relational import bureau, credit_card, installments, pos_cash, previous
from riskpilot.features.relational.common import FLOAT, ID, specs_to_catalog

logger = logging.getLogger(__name__)

SOURCE_MODULES: dict[str, ModuleType] = {
    "bureau": bureau,
    "previous": previous,
    "installments": installments,
    "credit_card": credit_card,
    "pos": pos_cash,
}
SOURCES: tuple[str, ...] = tuple(SOURCE_MODULES)
assert SOURCES == config.RELATIONAL_SOURCES

__all__ = [
    "SOURCES",
    "SOURCE_MODULES",
    "build_feature_matrix",
    "count_like_columns",
    "feature_catalog",
    "feature_source",
    "has_column",
    "load_source_table",
    "sources_of",
    "table_path",
]


def table_path(source: str, processed_dir: Path = config.RELATIONAL_PROCESSED_DIR) -> Path:
    return Path(processed_dir) / f"{source}.parquet"


def load_source_table(
    source: str, processed_dir: Path = config.RELATIONAL_PROCESSED_DIR
) -> pd.DataFrame:
    """Load a cached customer-level table (built by ``riskpilot.features.relational.build``)."""
    path = table_path(source, processed_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Build it with 'python -m riskpilot.features.relational.build "
            f"--sources {source}'."
        )
    return pd.read_parquet(path)


def has_column(source: str) -> str:
    return f"{source}__has_history"


def count_like_columns(source: str) -> list[str]:
    module = SOURCE_MODULES[source]
    return [f"{module.PREFIX}__{s.name}" for s in module.SPECS if s.count_like]


def feature_catalog(sources: Sequence[str] | None = None) -> pd.DataFrame:
    """Machine-readable catalog: one row per relational feature (+ history indicators)."""
    frames = []
    for source in sources or SOURCES:
        module = SOURCE_MODULES[source]
        frames.append(
            specs_to_catalog(
                module.SPECS, source=source, prefix=module.PREFIX, leakage=module.LEAKAGE
            )
        )
        frames.append(
            pd.DataFrame(
                [
                    {
                        "feature": has_column(source),
                        "source": source,
                        "family": "history indicator",
                        "description": f"1 if the customer has a row in {source}, else 0",
                        "aggregation": "indicator",
                        "count_like": True,
                        "leakage_assessment": "safe: presence of historical records only",
                    }
                ]
            )
        )
    return pd.concat(frames, ignore_index=True)


def feature_source(column: str) -> str:
    """Map a feature name to its source (``application`` when it has no known prefix)."""
    prefix = column.split("__", 1)[0] if "__" in column else None
    for source, module in SOURCE_MODULES.items():
        if prefix == module.PREFIX:
            return source
    return "application"


def sources_of(columns: Sequence[str]) -> pd.Series:
    return pd.Series([feature_source(c) for c in columns], index=list(columns), name="source")


def _check_table(source: str, table: pd.DataFrame) -> None:
    if ID not in table.columns:
        raise ValueError(f"{source}: customer table lacks {ID}.")
    if not table[ID].is_unique:
        raise ValueError(
            f"{source}: customer table is not one row per {ID} "
            f"({int(table[ID].duplicated().sum())} duplicates)."
        )
    prefix = SOURCE_MODULES[source].PREFIX + "__"
    bad = [c for c in table.columns if c != ID and not c.startswith(prefix)]
    if bad:
        raise ValueError(f"{source}: unexpected columns without the source prefix: {bad[:5]}")
    if config.TARGET_COL in table.columns:
        raise ValueError(f"{source}: TARGET must never be part of a feature table.")


def build_feature_matrix(
    X: pd.DataFrame,
    ids: pd.Series,
    sources: Sequence[str] = SOURCES,
    *,
    processed_dir: Path = config.RELATIONAL_PROCESSED_DIR,
    tables: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """``X`` (application features) + relational features of ``sources``, aligned on ``ids``.

    ``ids`` must be the ``SK_ID_CURR`` of each row of ``X`` (same length and
    order). ``tables`` can supply already-loaded customer tables (tests,
    notebooks); otherwise they are read from ``processed_dir``.
    """
    if len(ids) != len(X):
        raise ValueError(f"ids ({len(ids)}) and X ({len(X)}) must have the same length.")
    if not pd.Index(ids).is_unique:
        raise ValueError("Application identifiers must be unique before joining history.")
    forbidden = [c for c in (config.TARGET_COL, ID) if c in X.columns]
    if forbidden:
        raise ValueError(f"{forbidden} must not be part of the feature frame.")
    unknown = [s for s in sources if s not in SOURCE_MODULES]
    if unknown:
        raise ValueError(f"Unknown relational sources: {unknown}. Known: {list(SOURCE_MODULES)}")

    base = pd.DataFrame({ID: np.asarray(ids)}, index=X.index)
    blocks: list[pd.DataFrame] = [X]
    for source in sources:
        table = tables[source] if tables is not None and source in tables else None
        if table is None:
            table = load_source_table(source, processed_dir)
        _check_table(source, table)
        merged = base.merge(table, on=ID, how="left", validate="one_to_one", indicator=True)
        if len(merged) != len(X):
            raise AssertionError(
                f"{source}: join changed the row count ({len(merged)} vs {len(X)})."
            )
        merged.index = X.index
        has = (merged["_merge"] == "both").astype(FLOAT)
        block = merged.drop(columns=[ID, "_merge"])
        for col in count_like_columns(source):
            if col in block.columns:
                block[col] = block[col].where(has == 1.0, 0.0).astype(FLOAT)
        block[has_column(source)] = has
        blocks.append(block)
        logger.info(
            "%s: %d features joined, %.1f%% of rows have history",
            source,
            block.shape[1],
            100.0 * float(has.mean()),
        )
    out = pd.concat(blocks, axis=1)
    if len(out) != len(X) or not out.index.equals(X.index):
        raise AssertionError("Feature assembly changed the row count or the index.")
    if out.columns.duplicated().any():
        dupes = out.columns[out.columns.duplicated()][:5].tolist()
        raise AssertionError(f"Duplicate feature names after assembly: {dupes}")
    return out
