from pathlib import Path

import pandas as pd
import pytest

from riskpilot.data.load import load_application_train, split_features_target


def test_missing_file_raises_informative_error(tmp_path: Path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_application_train(missing)
    message = str(excinfo.value)
    assert str(missing) in message
    assert "kaggle competitions download" in message


def test_empty_file_raises(tmp_path: Path):
    empty = tmp_path / "application_train.csv"
    empty.touch()
    with pytest.raises(ValueError, match="empty"):
        load_application_train(empty)


def test_loads_small_fixture(synthetic_csv: Path, synthetic_df: pd.DataFrame):
    df = load_application_train(synthetic_csv)
    assert df.shape == synthetic_df.shape
    assert "TARGET" in df.columns and "SK_ID_CURR" in df.columns
    assert df["TARGET"].isin([0, 1]).all()


def test_nrows_and_usecols(synthetic_csv: Path):
    df = load_application_train(synthetic_csv, nrows=25, usecols=["AMT_INCOME_TOTAL"])
    assert len(df) == 25
    # Required columns are always included, even when not requested explicitly.
    assert set(df.columns) == {"SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL"}


def test_missing_required_column_raises(tmp_path: Path, synthetic_df: pd.DataFrame):
    path = tmp_path / "no_target.csv"
    synthetic_df.drop(columns=["TARGET"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="TARGET"):
        load_application_train(path)


def test_split_features_target_does_not_mutate(synthetic_df: pd.DataFrame):
    before = synthetic_df.copy()
    X, y = split_features_target(synthetic_df)
    pd.testing.assert_frame_equal(synthetic_df, before)
    assert "TARGET" not in X.columns
    assert "SK_ID_CURR" not in X.columns
    assert len(X) == len(y) == len(synthetic_df)
    assert y.name == "TARGET"


def test_split_features_target_requires_target(synthetic_df: pd.DataFrame):
    with pytest.raises(KeyError):
        split_features_target(synthetic_df.drop(columns=["TARGET"]))
