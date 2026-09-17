"""Feature preprocessing: one branch per model family.

* :mod:`riskpilot.features.preprocessing` - the linear (Logistic Regression)
  pipeline: impute + indicators + standardize + one-hot.
* :mod:`riskpilot.features.tree_preprocessing` - the gradient-boosting pipeline:
  native missing values, pandas categoricals, no scaling.
"""

from riskpilot.features.preprocessing import (
    build_baseline_pipeline,
    build_preprocessor,
    clean_features,
    infer_feature_types,
    replace_sentinels,
)
from riskpilot.features.tree_preprocessing import (
    TreePreprocessor,
    duplicate_missing_indicator_groups,
    select_heavy_tail_columns,
)

__all__ = [
    "TreePreprocessor",
    "build_baseline_pipeline",
    "build_preprocessor",
    "clean_features",
    "duplicate_missing_indicator_groups",
    "infer_feature_types",
    "replace_sentinels",
    "select_heavy_tail_columns",
]
