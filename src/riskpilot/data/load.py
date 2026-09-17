"""Loading the Home Credit ``application_train`` table.

The loader is intentionally thin: it locates the file, fails loudly when it is
absent, reads it with pandas and checks that the columns the rest of the
project relies on are present. It never mutates or caches anything.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from riskpilot import config

logger = logging.getLogger(__name__)

KAGGLE_DOWNLOAD_HINT = (
    "Download it with the official Kaggle CLI (requires Kaggle authentication and "
    "acceptance of the competition rules on kaggle.com):\n"
    f"    kaggle competitions download {config.KAGGLE_COMPETITION} "
    f"-f {config.APPLICATION_TRAIN_FILENAME} -p data/raw\n"
    "If Kaggle delivers a .zip archive, extract it so that the CSV sits at "
    f"{config.APPLICATION_TRAIN_PATH.relative_to(config.PROJECT_ROOT).as_posix()}."
)


def load_application_train(
    path: str | Path | None = None,
    *,
    nrows: int | None = None,
    usecols: Sequence[str] | None = None,
    required_columns: Iterable[str] = config.REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """Read ``application_train.csv`` into a DataFrame.

    Parameters
    ----------
    path:
        Location of the CSV. Defaults to ``config.APPLICATION_TRAIN_PATH``.
    nrows:
        Optional row limit, useful for smoke tests.
    usecols:
        Optional column subset. Required columns are always added.
    required_columns:
        Columns that must exist; a ``ValueError`` is raised otherwise.

    Raises
    ------
    FileNotFoundError
        If the file does not exist (with instructions on how to obtain it).
    ValueError
        If the file is empty or a required column is missing.
    """
    csv_path = Path(path) if path is not None else config.APPLICATION_TRAIN_PATH
    required = list(dict.fromkeys(required_columns))

    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found at '{csv_path}'.\n{KAGGLE_DOWNLOAD_HINT}")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"Dataset file '{csv_path}' is empty.")

    if usecols is not None:
        usecols = list(dict.fromkeys([*required, *usecols]))

    df = pd.read_csv(csv_path, nrows=nrows, usecols=usecols, low_memory=False)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{csv_path}' is missing required column(s): {missing}. "
            f"Found {len(df.columns)} columns."
        )

    logger.info("Loaded %s: %d rows x %d columns", csv_path.name, len(df), df.shape[1])
    return df


def categorize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with object (string) columns stored as ``pandas.Categorical``.

    Pure memory measure: the 16 string columns of ``application_train`` hold
    about 330 MB as Python objects and about 5 MB as categoricals, which halves
    the in-memory table. Values are unchanged, missing stays missing, and every
    downstream step (the linear pipeline's imputer + one-hot encoder, the tree
    preprocessor) treats a categorical column exactly like a string column; the
    challenger experiment verifies this by reproducing the recorded baseline
    metrics from the persisted pipeline.
    """
    out = df.copy(deep=False)
    string_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        out[col] = out[col].astype("category")
    return out


def split_features_target(
    df: pd.DataFrame,
    *,
    target_col: str = config.TARGET_COL,
    drop_cols: Sequence[str] = (config.ID_COL,),
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target without mutating ``df``.

    Identifier columns listed in ``drop_cols`` are removed from the features:
    ``SK_ID_CURR`` is a row key, not a predictor, and keeping it would be a
    classic leakage / overfitting vector for tree models later on.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    to_drop = [target_col, *[c for c in drop_cols if c in df.columns]]
    X = df.drop(columns=to_drop)
    y = df[target_col].copy()
    return X, y
