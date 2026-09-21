"""Customer-level features from the Home Credit relational (historical) tables.

One module per source (``bureau`` incl. ``bureau_balance``, ``previous``,
``installments``, ``credit_card``, ``pos_cash``), each producing one row per
``SK_ID_CURR`` with prefixed ``float32`` features and a documented
:class:`~riskpilot.features.relational.common.FeatureSpec` per column;
``assemble`` joins them onto the application rows with validated one-to-one
merges; ``build`` is the cached command-line builder. ``build`` is a CLI module
and is not imported here (see ``riskpilot.models.__init__`` for the reason).
"""

from riskpilot.features.relational.assemble import (
    SOURCES,
    build_feature_matrix,
    feature_catalog,
    feature_source,
    load_source_table,
    sources_of,
)
from riskpilot.features.relational.common import FeatureSpec

__all__ = [
    "SOURCES",
    "FeatureSpec",
    "build_feature_matrix",
    "feature_catalog",
    "feature_source",
    "load_source_table",
    "sources_of",
]
