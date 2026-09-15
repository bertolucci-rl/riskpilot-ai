import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from riskpilot.data.load import split_features_target
from riskpilot.data.validation import FeatureTypes
from riskpilot.features.preprocessing import (
    build_baseline_pipeline,
    build_preprocessor,
    clean_features,
    infer_feature_types,
    make_sentinel_transformer,
    replace_sentinels,
)


@pytest.fixture
def xy(synthetic_df):
    return split_features_target(synthetic_df)


def test_infer_feature_types(xy):
    X, _ = xy
    types = infer_feature_types(X)
    assert set(types.categorical) == {"NAME_CONTRACT_TYPE", "CODE_GENDER", "OCCUPATION_TYPE"}
    assert set(types.numeric) == set(X.columns) - set(types.categorical)


def test_replace_sentinels_is_pure(xy):
    X, _ = xy
    before = X.copy()
    out = replace_sentinels(X, {"DAYS_EMPLOYED": 365243.0})
    pd.testing.assert_frame_equal(X, before)  # no hidden mutation
    assert (X["DAYS_EMPLOYED"] == 365243).sum() > 0
    assert (out["DAYS_EMPLOYED"] == 365243).sum() == 0
    assert out["DAYS_EMPLOYED"].isna().sum() == (X["DAYS_EMPLOYED"] == 365243).sum()
    # Columns not present are ignored silently.
    replace_sentinels(X, {"NOT_A_COLUMN": 1.0})


def test_sentinel_transformer_keeps_dataframe(xy):
    X, _ = xy
    out = make_sentinel_transformer({"DAYS_EMPLOYED": 365243.0}).fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(X.columns)
    # Python None in object columns is canonicalized to np.nan for scikit-learn.
    assert X["OCCUPATION_TYPE"].isna().sum() > 0
    assert not any(v is None for v in out["OCCUPATION_TYPE"])
    assert out["OCCUPATION_TYPE"].isna().sum() == X["OCCUPATION_TYPE"].isna().sum()


def test_build_preprocessor_structure(xy):
    X, _ = xy
    types = infer_feature_types(X)
    pre = build_preprocessor(types)
    assert isinstance(pre, ColumnTransformer)
    assert [name for name, _, _ in pre.transformers] == ["num", "cat"]


def test_build_preprocessor_requires_features():
    with pytest.raises(ValueError):
        build_preprocessor(FeatureTypes(numeric=[], categorical=[]))


def test_preprocessor_fit_transform_shape(xy):
    X, _ = xy
    X = clean_features(X)  # the ColumnTransformer expects canonicalized input
    types = infer_feature_types(X)
    pre = build_preprocessor(types)
    Xt = pre.fit_transform(X)
    names = pre.get_feature_names_out()
    assert Xt.shape == (len(X), len(names))
    # Missing indicators exist only for numeric columns that had NaN in training.
    assert "num__missingindicator_EXT_SOURCE_1" in names
    assert "num__missingindicator_AMT_INCOME_TOTAL" not in names
    # Missing categoricals become their own level.
    assert "cat__OCCUPATION_TYPE_missing" in names
    # Numeric branch is standardized.
    dense = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    num_block = dense[:, : len(types.numeric)]
    assert np.allclose(num_block.mean(axis=0), 0.0, atol=1e-8)


def test_unknown_categories_are_ignored_at_transform_time(xy):
    X, _ = xy
    types = infer_feature_types(X)
    pre = build_preprocessor(types).fit(X)
    new = X.head(3).copy()
    new.loc[new.index[0], "OCCUPATION_TYPE"] = "Astronaut"  # unseen level
    new.loc[new.index[1], "CODE_GENDER"] = np.nan
    Xt = pre.transform(new)
    dense = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    names = list(pre.get_feature_names_out())
    occ_cols = [i for i, n in enumerate(names) if n.startswith("cat__OCCUPATION_TYPE_")]
    assert Xt.shape[1] == len(names)
    assert dense[0, occ_cols].sum() == 0.0  # unseen level -> all-zero one-hot block


def test_baseline_pipeline_fits_and_predicts_probabilities(xy):
    X, y = xy
    pipe = build_baseline_pipeline(infer_feature_types(X), max_iter=500)
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["clean", "preprocess", "model"]
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))
    assert pipe["model"].class_weight is None
    # The fitted pipeline never saw the sentinel as a magnitude.
    assert pipe["model"].n_iter_[0] < pipe["model"].max_iter


def test_pipeline_handles_unseen_rows_without_error(xy):
    X, y = xy
    pipe = build_baseline_pipeline(infer_feature_types(X), max_iter=500).fit(X, y)
    row = X.head(1).copy()
    row["OCCUPATION_TYPE"] = "Astronaut"
    row["EXT_SOURCE_1"] = np.nan
    row["DAYS_EMPLOYED"] = 365243.0
    proba = pipe.predict_proba(row)[:, 1]
    assert 0.0 <= proba[0] <= 1.0
