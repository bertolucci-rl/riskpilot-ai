"""Centralized paths and constants for RiskPilot AI.

Every path is derived from the location of this file, so the package works from
any working directory and on any operating system (including Windows). No
user-specific absolute path may appear anywhere in the repository.
"""

from __future__ import annotations

from pathlib import Path

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
