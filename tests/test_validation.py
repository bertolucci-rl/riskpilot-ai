import json

import numpy as np
import pandas as pd
import pytest

from riskpilot.data import validation as v


def test_validate_target_accepts_binary(synthetic_df):
    v.validate_target(synthetic_df["TARGET"])


@pytest.mark.parametrize(
    "series, match",
    [
        (pd.Series([0, 1, None]), "missing"),
        (pd.Series([0, 1, 2]), "unexpected"),
        (pd.Series([0, 0, 0]), "single class"),
    ],
)
def test_validate_target_rejects_bad_targets(series, match):
    with pytest.raises(ValueError, match=match):
        v.validate_target(series)


def test_summarize_target_prevalence():
    y = pd.Series([0, 0, 0, 1])
    summary = v.summarize_target(y)
    assert summary["n"] == 4
    assert summary["n_positive"] == 1
    assert summary["prevalence"] == pytest.approx(0.25)
    assert summary["imbalance_ratio"] == pytest.approx(3.0)


def test_dataset_overview_detects_duplicates(synthetic_df):
    dup = pd.concat([synthetic_df, synthetic_df.iloc[:3]], ignore_index=True)
    overview = v.dataset_overview(dup)
    assert overview["n_duplicate_rows"] == 3
    assert overview["n_duplicate_ids"] == 3
    assert overview["id_is_unique"] is False
    assert v.dataset_overview(synthetic_df)["id_is_unique"] is True


def test_split_feature_types(synthetic_df):
    types = v.split_feature_types(synthetic_df)
    assert "TARGET" not in types.all and "SK_ID_CURR" not in types.all
    assert set(types.categorical) == {"NAME_CONTRACT_TYPE", "CODE_GENDER", "OCCUPATION_TYPE"}
    assert "AMT_INCOME_TOTAL" in types.numeric and "FLAG_MOBIL" in types.numeric


def test_split_feature_types_rejects_unsupported_dtype(synthetic_df):
    df = synthetic_df.assign(WHEN=pd.Timestamp("2020-01-01"))
    with pytest.raises(TypeError, match="WHEN"):
        v.split_feature_types(df)


def test_missingness_table_sorted(synthetic_df):
    table = v.missingness_table(synthetic_df)
    assert list(table["frac_missing"]) == sorted(table["frac_missing"], reverse=True)
    occ = table.set_index("column").loc["OCCUPATION_TYPE", "frac_missing"]
    assert 0.2 < occ < 0.4


def test_constant_features_flags_constant_column(synthetic_df):
    table = v.constant_features(synthetic_df)
    row = table.set_index("column").loc["FLAG_MOBIL"]
    assert bool(row["is_constant"]) is True
    assert row["top_frac"] == pytest.approx(1.0)


def test_high_cardinality_features():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": range(100),
            "TARGET": [0, 1] * 50,
            "LOW": ["a", "b"] * 50,
            "HIGH": [f"cat_{i}" for i in range(100)],
        }
    )
    table = v.high_cardinality_features(df, threshold=20)
    assert list(table["column"]) == ["HIGH"]


def test_suspicious_values_report(synthetic_df):
    report = v.suspicious_values_report(synthetic_df)
    assert "CODE_GENDER" in report["unknown_tokens"]
    assert report["unknown_tokens"]["CODE_GENDER"].get("XNA", 0) > 0
    assert "DAYS_EMPLOYED" in report["days_columns_positive"]
    sent = report["sentinels"]["DAYS_EMPLOYED"]
    assert sent["n"] == int((synthetic_df["DAYS_EMPLOYED"] == 365243).sum())
    assert sent["n"] > 0


def test_leakage_screen_flags_obvious_leak(synthetic_df):
    leaky = synthetic_df.assign(TARGET_COPY=synthetic_df["TARGET"] * 1.0)
    report = v.leakage_screen(leaky)
    assert "TARGET_COPY" in report["name_based_flags"]
    assert report["high_separation_features"]["TARGET_COPY"] == pytest.approx(1.0)
    # Genuine features in the fixture never reach perfect separation.
    assert all(
        auc < 1.0 for col, auc in report["top_univariate_auc"].items() if col != "TARGET_COPY"
    )


def test_run_validation_is_json_serializable(synthetic_df):
    report = v.run_validation(synthetic_df)
    encoded = json.dumps(report)
    decoded = json.loads(encoded)
    assert decoded["overview"]["n_rows"] == len(synthetic_df)
    assert decoded["target"]["prevalence"] == pytest.approx(synthetic_df["TARGET"].mean())
    assert decoded["feature_types"]["n_categorical"] == 3
    assert isinstance(decoded["leakage"]["top_univariate_auc"], dict)


def test_run_validation_requires_target(synthetic_df):
    with pytest.raises(ValueError, match="TARGET"):
        v.run_validation(synthetic_df.drop(columns=["TARGET"]))


def test_py_converts_numpy_scalars():
    assert isinstance(v._py(np.int64(3)), int)
    assert isinstance(v._py(np.float32(1.5)), float)
    assert v._py("x") == "x"
