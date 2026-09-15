"""Shared fixtures: a small synthetic frame shaped like ``application_train``.

The synthetic data mimics the structural quirks that the pipeline must handle:
numeric columns with missing values, a sentinel-encoded ``DAYS_EMPLOYED``,
categorical columns with missing values and an "XNA" token, a constant column
and an imbalanced binary target that is (weakly) related to the features.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # tests never need an interactive GUI backend

N_ROWS = 400


def make_synthetic_application(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ext_source = rng.uniform(0, 1, n)
    income = rng.lognormal(mean=11.5, sigma=0.5, size=n)
    days_birth = -rng.integers(20 * 365, 65 * 365, n)
    days_employed = -rng.integers(0, 30 * 365, n).astype(float)
    sentinel_mask = rng.uniform(size=n) < 0.15
    days_employed[sentinel_mask] = 365243.0

    contract = rng.choice(["Cash loans", "Revolving loans"], size=n, p=[0.9, 0.1])
    gender = rng.choice(["F", "M", "XNA"], size=n, p=[0.65, 0.34, 0.01])
    occupation = rng.choice(
        ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", None],
        size=n,
        p=[0.25, 0.15, 0.15, 0.1, 0.05, 0.3],
    )

    # Latent default risk: lower EXT_SOURCE and younger applicants default more.
    logit = -2.6 - 2.5 * (ext_source - 0.5) + 0.4 * (days_birth / 365 + 40) / 20
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.uniform(size=n) < prob).astype(int)

    ext_source_missing = ext_source.copy()
    ext_source_missing[rng.uniform(size=n) < 0.2] = np.nan

    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + n),
            "TARGET": target,
            "NAME_CONTRACT_TYPE": contract,
            "CODE_GENDER": gender,
            "OCCUPATION_TYPE": occupation,
            "AMT_INCOME_TOTAL": income,
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "EXT_SOURCE_1": ext_source_missing,
            "FLAG_MOBIL": np.ones(n, dtype=int),  # constant column
            "CNT_CHILDREN": rng.poisson(0.4, n),
        }
    )


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_application()


@pytest.fixture
def synthetic_csv(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    path = tmp_path / "application_train.csv"
    synthetic_df.to_csv(path, index=False)
    return path
