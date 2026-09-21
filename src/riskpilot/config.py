"""Centralized paths and constants for RiskPilot AI.

Every path is derived from the location of this file, so the package works from
any working directory and on any operating system (including Windows). No
user-specific absolute path may appear anywhere in the repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

KAGGLE_COMPETITION: str = "home-credit-default-risk"
APPLICATION_TRAIN_FILENAME: str = "application_train.csv"
APPLICATION_TRAIN_PATH: Path = RAW_DATA_DIR / APPLICATION_TRAIN_FILENAME
COLUMNS_DESCRIPTION_FILENAME: str = "HomeCredit_columns_description.csv"

# Relational (historical) tables of the competition, keyed by the short source
# name used throughout ``riskpilot.features.relational``. ``keys`` are the
# columns every download must carry; ``grain`` documents the expected row
# identity (verified against the data in the relational audit).
RELATIONAL_TABLES: dict[str, dict[str, Any]] = {
    "bureau": {
        "filename": "bureau.csv",
        "keys": ("SK_ID_CURR", "SK_ID_BUREAU"),
        "grain": "one row per external credit (SK_ID_BUREAU), many per SK_ID_CURR",
    },
    "bureau_balance": {
        "filename": "bureau_balance.csv",
        "keys": ("SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"),
        "grain": "one row per external credit and month (SK_ID_BUREAU, MONTHS_BALANCE)",
    },
    "previous": {
        "filename": "previous_application.csv",
        "keys": ("SK_ID_PREV", "SK_ID_CURR"),
        "grain": "one row per previous Home Credit application (SK_ID_PREV), many per SK_ID_CURR",
    },
    "installments": {
        "filename": "installments_payments.csv",
        "keys": ("SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_NUMBER"),
        "grain": "one row per (previous credit, installment, payment); split payments repeat",
    },
    "credit_card": {
        "filename": "credit_card_balance.csv",
        "keys": ("SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE"),
        "grain": "one row per previous credit card and month (SK_ID_PREV, MONTHS_BALANCE)",
    },
    "pos": {
        "filename": "POS_CASH_balance.csv",
        "keys": ("SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE"),
        "grain": "one row per previous POS/cash credit and month (SK_ID_PREV, MONTHS_BALANCE)",
    },
}
RELATIONAL_SOURCES: tuple[str, ...] = ("bureau", "previous", "installments", "credit_card", "pos")
RELATIONAL_PROCESSED_DIR: Path = PROCESSED_DATA_DIR / "relational"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
# Share of the frozen training portion held out for model selection / early
# stopping of the challengers (Milestone 2). The frozen test split is never used.
VALIDATION_SIZE: float = 0.2
# Threads for the gradient-boosting libraries: the physical core count of the
# development machine. Recorded in every metrics file; override with --n-jobs.
N_JOBS: int = 6

TARGET_COL: str = "TARGET"
ID_COL: str = "SK_ID_CURR"
REQUIRED_COLUMNS: tuple[str, ...] = (ID_COL, TARGET_COL)
ALLOWED_TARGET_VALUES: tuple[int, ...] = (0, 1)

# Sentinel encodings in the Home Credit application table that must be treated
# as "missing" rather than as real magnitudes. DAYS_EMPLOYED uses 365243
# (roughly +1000 years) for applicants without an employment record
# (pensioners / unemployed). The preprocessing pipeline maps these to NaN so the
# median imputer + missing indicator handle them. The presence of the sentinel
# is verified empirically in ``riskpilot.data.validation`` and in notebook 01.
SENTINEL_VALUES: dict[str, float] = {"DAYS_EMPLOYED": 365243.0}
