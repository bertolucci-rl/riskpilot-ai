import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from riskpilot import config
from riskpilot.data.load import split_features_target
from riskpilot.features.tree_preprocessing import (
    TreePreprocessor,
    duplicate_missing_indicator_groups,
    select_heavy_tail_columns,
)


@pytest.fixture
def xy(synthetic_df):
    return split_features_target(synthetic_df)


def test_fit_transform_keeps_raw_scale_and_native_types(xy):
    X, _ = xy
    pre = TreePreprocessor().fit(X)
    Xt = pre.transform(X)

    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape == (len(X), X.shape[1])  # no dimensional blow-up: one column per feature
    assert list(Xt.columns) == list(pre.get_feature_names_out())
    numeric = pre.numeric_columns_
    categorical = pre.categorical_columns_
    assert set(categorical) == {"NAME_CONTRACT_TYPE", "CODE_GENDER", "OCCUPATION_TYPE"}
    assert all(Xt[c].dtype == np.float32 for c in numeric)
    assert all(isinstance(Xt[c].dtype, pd.CategoricalDtype) for c in categorical)
    # No scaling and no imputation: values are the raw ones, missing stays missing.
    np.testing.assert_allclose(Xt["AMT_INCOME_TOTAL"], X["AMT_INCOME_TOTAL"].astype(np.float32))
    assert Xt["EXT_SOURCE_1"].isna().sum() == X["EXT_SOURCE_1"].isna().sum()
    assert abs(Xt["DAYS_BIRTH"].mean()) > 1.0  # not standardized


def test_sentinel_is_mapped_to_missing(xy):
    X, _ = xy
    Xt = TreePreprocessor().fit_transform(X)
    n_sentinel = int((X["DAYS_EMPLOYED"] == config.SENTINEL_VALUES["DAYS_EMPLOYED"]).sum())
    assert n_sentinel > 0
    assert Xt["DAYS_EMPLOYED"].isna().sum() == n_sentinel
    assert (Xt["DAYS_EMPLOYED"].dropna() <= 0).all()


def test_categories_are_learned_on_training_data_only(xy):
    X, _ = xy
    train, new = X.iloc[:300], X.iloc[300:].copy()
    pre = TreePreprocessor().fit(train)
    new.loc[new.index[0], "OCCUPATION_TYPE"] = "Astronaut"  # unseen level
    Xt_train, Xt_new = pre.transform(train), pre.transform(new)

    assert list(Xt_new["OCCUPATION_TYPE"].cat.categories) == pre.categories_["OCCUPATION_TYPE"]
    assert list(Xt_new["OCCUPATION_TYPE"].cat.categories) == list(
        Xt_train["OCCUPATION_TYPE"].cat.categories
    )
    assert pd.isna(Xt_new["OCCUPATION_TYPE"].iloc[0])  # unseen -> missing, never a new code
    # Same value -> same code at fit and at transform time (what the boosters rely on).
    code_train = Xt_train["CODE_GENDER"].cat.codes[Xt_train["CODE_GENDER"] == "F"].iloc[0]
    code_new = Xt_new["CODE_GENDER"].cat.codes[Xt_new["CODE_GENDER"] == "F"].iloc[0]
    assert code_train == code_new
    # Missing categoricals stay missing rather than becoming a level.
    assert Xt_train["OCCUPATION_TYPE"].isna().sum() == train["OCCUPATION_TYPE"].isna().sum()


def test_refuses_target_and_identifier_columns(synthetic_df, xy):
    X, _ = xy
    with pytest.raises(ValueError, match="TARGET"):
        TreePreprocessor().fit(synthetic_df)
    pre = TreePreprocessor().fit(X)
    with pytest.raises(ValueError, match="SK_ID_CURR"):
        pre.transform(synthetic_df.drop(columns=["TARGET"]))
    with pytest.raises(ValueError, match="missing feature columns"):
        pre.transform(X.drop(columns=["EXT_SOURCE_1"]))


def test_heavy_tail_selection_is_data_driven(xy):
    X, _ = xy
    selected = select_heavy_tail_columns(X)
    # Log-normal income is skewed, non-negative and unbounded; negative DAYS_*,
    # the [0, 1] score, the constant flag and the small count are excluded.
    assert selected == ["AMT_INCOME_TOTAL"]
    assert select_heavy_tail_columns(X, min_skew=1e9) == []


def test_log1p_treatment_applies_to_selected_columns_only(xy):
    X, _ = xy
    pre = TreePreprocessor(heavy_tail="log1p").fit(X)
    Xt = pre.transform(X)
    assert pre.heavy_tail_columns_ == ["AMT_INCOME_TOTAL"]
    np.testing.assert_allclose(
        Xt["AMT_INCOME_TOTAL"], np.log1p(X["AMT_INCOME_TOTAL"]).astype(np.float32), rtol=1e-6
    )
    np.testing.assert_allclose(Xt["DAYS_BIRTH"], X["DAYS_BIRTH"].astype(np.float32))


def test_winsorization_bounds_come_from_training_data_only(xy):
    X, _ = xy
    pre = TreePreprocessor(heavy_tail="winsorize", winsor_quantiles=(0.01, 0.99)).fit(X)
    lo, hi = pre.clip_bounds_["AMT_INCOME_TOTAL"]
    assert lo == pytest.approx(X["AMT_INCOME_TOTAL"].quantile(0.01))
    assert hi == pytest.approx(X["AMT_INCOME_TOTAL"].quantile(0.99))
    new = X.head(2).copy()
    new.loc[new.index[0], "AMT_INCOME_TOTAL"] = 1e12  # an outlier never seen in training
    new.loc[new.index[1], "AMT_INCOME_TOTAL"] = 0.0
    Xt = pre.transform(new)
    assert Xt["AMT_INCOME_TOTAL"].iloc[0] == pytest.approx(hi, rel=1e-6)
    assert Xt["AMT_INCOME_TOTAL"].iloc[1] == pytest.approx(lo, rel=1e-6)


def test_explicit_heavy_tail_columns_are_validated(xy):
    X, _ = xy
    with pytest.raises(ValueError, match="not numeric or not present"):
        TreePreprocessor(heavy_tail="log1p", heavy_tail_columns=["NOT_A_COLUMN"]).fit(X)
    pre = TreePreprocessor(heavy_tail="log1p", heavy_tail_columns=["CNT_CHILDREN"]).fit(X)
    assert pre.heavy_tail_columns_ == ["CNT_CHILDREN"]


def test_missing_indicators_all_vs_deduplicated(xy):
    X, _ = xy
    X = X.assign(EXT_SOURCE_COPY=X["EXT_SOURCE_1"] * 2)  # identical missingness pattern

    groups = duplicate_missing_indicator_groups(
        X, ["EXT_SOURCE_1", "EXT_SOURCE_COPY", "DAYS_BIRTH"]
    )
    assert ["EXT_SOURCE_1", "EXT_SOURCE_COPY"] in groups
    assert ["DAYS_BIRTH"] in groups

    none = TreePreprocessor().fit(X)
    full = TreePreprocessor(missing_indicators="all").fit(X)
    dedup = TreePreprocessor(missing_indicators="deduplicated").fit(X)
    assert none.indicator_columns_ == []
    assert set(full.indicator_columns_) == {"EXT_SOURCE_1", "EXT_SOURCE_COPY", "DAYS_EMPLOYED"}
    assert set(dedup.indicator_columns_) == {"EXT_SOURCE_1", "DAYS_EMPLOYED"}
    assert ["EXT_SOURCE_1", "EXT_SOURCE_COPY"] in dedup.indicator_groups_

    Xt = full.transform(X)
    assert Xt.shape[1] == X.shape[1] + 3
    assert Xt["missing_EXT_SOURCE_1"].dtype == np.float32
    np.testing.assert_array_equal(
        Xt["missing_EXT_SOURCE_1"], X["EXT_SOURCE_1"].isna().astype(np.float32)
    )
    # The de-duplicated indicator carries exactly the same information as the dropped one.
    np.testing.assert_array_equal(Xt["missing_EXT_SOURCE_1"], Xt["missing_EXT_SOURCE_COPY"])
    assert dedup.transform(X).shape[1] == X.shape[1] + 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"heavy_tail": "sqrt"},
        {"missing_indicators": "some"},
        {"winsor_quantiles": (0.9, 0.1)},
    ],
)
def test_invalid_parameters_raise(xy, kwargs):
    X, _ = xy
    with pytest.raises(ValueError):
        TreePreprocessor(**kwargs).fit(X)


def test_transform_before_fit_raises(xy):
    X, _ = xy
    with pytest.raises(Exception):  # noqa: B017 - scikit-learn's NotFittedError
        TreePreprocessor().transform(X)


def test_describe_is_json_serializable_and_clone_safe(xy):
    X, _ = xy
    pre = TreePreprocessor(heavy_tail="winsorize", missing_indicators="deduplicated")
    cloned = clone(pre)
    assert cloned.get_params() == pre.get_params()
    pre.fit(X)
    desc = pre.describe()
    json.dumps(desc)
    assert desc["n_features_out"] == pre.get_feature_names_out().shape[0]
    assert desc["heavy_tail"] == "winsorize" and desc["missing_indicators"] == "deduplicated"
    assert desc["categorical_cardinality"]["CODE_GENDER"] == 3
