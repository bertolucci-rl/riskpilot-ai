import json

import numpy as np
import pandas as pd
import pytest

from riskpilot import config
from riskpilot.data.load import split_features_target
from riskpilot.models import comparison as cmp
from riskpilot.models.evaluate import save_metrics
from riskpilot.models.train import BaselineConfig, make_split, run_baseline


def _membership(synthetic_df, path, *, random_state=42):
    X, y = split_features_target(synthetic_df)
    _, X_test, _, _ = make_split(X, y, random_state=random_state)
    frame = pd.DataFrame({config.ID_COL: synthetic_df[config.ID_COL], "split": "train"})
    frame.loc[X_test.index, "split"] = "test"
    frame.to_csv(path, index=False)
    return frame


@pytest.fixture
def parts(synthetic_df):
    X, y = split_features_target(synthetic_df)
    return X, y, synthetic_df[config.ID_COL]


# --------------------------------------------------------------------------- #
# Frozen split
# --------------------------------------------------------------------------- #
def test_load_frozen_split_matches_membership_and_seed(synthetic_df, parts, tmp_path):
    X, y, ids = parts
    membership = _membership(synthetic_df, tmp_path / "m.csv")
    split = cmp.load_frozen_split(X, y, ids, membership_path=tmp_path / "m.csv")

    expected_test = set(membership.loc[membership["split"] == "test", config.ID_COL])
    assert set(split.ids_test) == expected_test
    assert set(split.ids_train).isdisjoint(split.ids_test)
    assert len(split.y_train) + len(split.y_test) == len(y)
    assert split.verified_against_seed is True
    assert split.summary()["n_test"] == len(expected_test)
    pd.testing.assert_frame_equal(split.X_test, X.loc[split.X_test.index])


def test_load_frozen_split_rejects_a_different_holdout(synthetic_df, parts, tmp_path):
    X, y, ids = parts
    _membership(synthetic_df, tmp_path / "other.csv", random_state=7)  # not the frozen split
    with pytest.raises(ValueError, match="does not reproduce"):
        cmp.load_frozen_split(X, y, ids, membership_path=tmp_path / "other.csv")
    # Without verification the file is taken as-is.
    split = cmp.load_frozen_split(X, y, ids, membership_path=tmp_path / "other.csv", verify=False)
    assert split.verified_against_seed is False


def test_load_frozen_split_rejects_mismatched_ids_and_bad_files(synthetic_df, parts, tmp_path):
    X, y, ids = parts
    with pytest.raises(FileNotFoundError):
        cmp.load_frozen_split(X, y, ids, membership_path=tmp_path / "missing.csv")
    _membership(synthetic_df, tmp_path / "m.csv")
    with pytest.raises(ValueError, match="do not match"):
        cmp.load_frozen_split(
            X.iloc[:100], y.iloc[:100], ids.iloc[:100], membership_path=tmp_path / "m.csv"
        )
    pd.DataFrame({"foo": [1], "split": ["test"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="must have columns"):
        cmp.load_frozen_split(X, y, ids, membership_path=tmp_path / "bad.csv")


# --------------------------------------------------------------------------- #
# Baseline reference (integrity of the Milestone 1 numbers)
# --------------------------------------------------------------------------- #
@pytest.fixture
def baseline_run(synthetic_csv, tmp_path):
    cfg = BaselineConfig(
        data_path=synthetic_csv,
        figures_dir=tmp_path / "figures",
        metrics_dir=tmp_path / "metrics",
        models_dir=tmp_path / "models",
        processed_dir=tmp_path / "processed",
        max_iter=500,
    )
    result = run_baseline(cfg)
    df = pd.read_csv(synthetic_csv)
    X, y = split_features_target(df)
    split = cmp.load_frozen_split(
        X,
        y,
        df[config.ID_COL],
        membership_path=tmp_path / "processed" / "baseline_split_membership.csv",
    )
    return result, split, cfg


def test_baseline_reference_reloads_model_and_matches_recorded_metrics(baseline_run):
    result, split, cfg = baseline_run
    ref = cmp.load_baseline_reference(
        split, model_path=result.model_path, metrics_path=result.metrics_path
    )
    assert ref.integrity["model_reloaded_from_disk"] is True
    assert ref.integrity["metrics_match_recorded"] is True
    assert ref.integrity["max_abs_metric_difference"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(np.sort(ref.y_prob_test), np.sort(result.y_prob_test))
    assert ref.summary["metrics_test"]["roc_auc"] == pytest.approx(
        result.payload["metrics"]["test"]["roc_auc"]
    )
    assert ref.summary["n_features"] == result.payload["data"]["n_transformed_features"]


def test_baseline_reference_refits_when_model_is_missing(baseline_run, tmp_path):
    result, split, cfg = baseline_run
    ref = cmp.load_baseline_reference(
        split, model_path=tmp_path / "absent.joblib", metrics_path=result.metrics_path
    )
    assert ref.integrity["model_reloaded_from_disk"] is False
    assert ref.integrity["metrics_match_recorded"] is True
    assert ref.summary["model_size_mb"] is None


def test_baseline_reference_refuses_drifted_metrics(baseline_run, tmp_path):
    result, split, cfg = baseline_run
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    payload["metrics"]["test"]["roc_auc"] += 0.05
    drifted = save_metrics(payload, tmp_path / "drifted.json")
    with pytest.raises(ValueError, match="does not reproduce"):
        cmp.load_baseline_reference(split, model_path=result.model_path, metrics_path=drifted)


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
def _summary(label, **metrics):
    base = {
        "roc_auc": 0.75,
        "average_precision": 0.23,
        "log_loss": 0.25,
        "brier_score": 0.068,
        "brier_skill_score": 0.08,
        "expected_calibration_error": 0.002,
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
        "mean_predicted_probability": 0.08,
        "max_predicted_probability": 0.7,
    }
    base.update(metrics)
    return {
        "label": label,
        "n_features": 10,
        "n_trees": None,
        "fit_seconds": 1.0,
        "predict_seconds_test": 0.1,
        "model_size_mb": 0.5,
        "metrics_test": base,
        "metrics_train": {"roc_auc": base["roc_auc"], "log_loss": base["log_loss"]},
    }


def test_comparison_table_computes_absolute_and_relative_deltas():
    summaries = {
        "logistic_regression": _summary("Logistic Regression"),
        "lightgbm": _summary("LightGBM", roc_auc=0.77, log_loss=0.24),
    }
    table = cmp.comparison_table(summaries)
    assert list(table["model"]) == ["Logistic Regression", "LightGBM"]
    ref, lgbm = table.iloc[0], table.iloc[1]
    assert ref["delta_roc_auc"] == 0.0 and ref["rel_log_loss_pct"] == 0.0
    assert lgbm["delta_roc_auc"] == pytest.approx(0.02)
    assert lgbm["delta_log_loss"] == pytest.approx(-0.01)
    assert lgbm["rel_log_loss_pct"] == pytest.approx(-4.0)
    assert {"fit_seconds", "n_features", "model_size_mb", "train_roc_auc"} <= set(table.columns)
    with pytest.raises(KeyError):
        cmp.comparison_table(summaries, reference="xgboost")


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #
@pytest.fixture
def scored():
    rng = np.random.default_rng(11)
    n = 1500
    logit = rng.normal(-2.5, 1.0, size=n)
    p_good = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p_good).astype(int)
    p_noisy = 1 / (1 + np.exp(-(0.5 * logit + rng.normal(0, 1.0, size=n) - 1.2)))
    return y, {"logistic_regression": p_noisy, "lightgbm": p_good, "xgboost": p_good.copy()}


def test_bootstrap_replicates_are_paired_and_deterministic(scored):
    y, preds = scored
    reps = cmp.bootstrap_replicates(y, preds, n_replicates=25, seed=3)
    assert set(reps) == set(preds)
    assert all(frame.shape == (25, len(cmp.BOOTSTRAP_METRICS)) for frame in reps.values())
    # Identical predictions -> identical metric paths (same resample for every model).
    pd.testing.assert_frame_equal(reps["lightgbm"], reps["xgboost"])
    again = cmp.bootstrap_replicates(y, preds, n_replicates=25, seed=3)
    pd.testing.assert_frame_equal(reps["lightgbm"], again["lightgbm"])
    other = cmp.bootstrap_replicates(y, preds, n_replicates=25, seed=4)
    assert not np.allclose(reps["lightgbm"]["roc_auc"], other["lightgbm"]["roc_auc"])


def test_paired_bootstrap_summaries(scored):
    y, preds = scored
    boot = cmp.paired_bootstrap(y, preds, n_replicates=60, seed=5)
    assert (
        boot["n_replicates"] == 60
        and boot["seed"] == 5
        and boot["reference"] == "logistic_regression"
    )
    assert set(boot["differences"]) == {
        "lightgbm_vs_logistic_regression",
        "xgboost_vs_logistic_regression",
        "xgboost_vs_lightgbm",
    }
    better = boot["differences"]["lightgbm_vs_logistic_regression"]
    for metric in cmp.BOOTSTRAP_METRICS:
        s = better[metric]
        assert s["ci_low"] <= s["ci_high"]
        assert 0.0 <= s["fraction_improving"] <= 1.0
    assert better["roc_auc"]["estimate"] > 0 and better["roc_auc"]["ci_excludes_zero"] is True
    assert better["log_loss"]["estimate"] < 0 and better["log_loss"]["fraction_improving"] > 0.95
    same = boot["differences"]["xgboost_vs_lightgbm"]
    for metric in cmp.BOOTSTRAP_METRICS:
        assert (
            same[metric]["estimate"] == 0.0
            and same[metric]["ci_low"] == 0.0 == same[metric]["ci_high"]
        )
        assert same[metric]["ci_excludes_zero"] is False
    for name, p in preds.items():
        assert boot["models"][name]["roc_auc"]["estimate"] == pytest.approx(cmp.roc_auc_score(y, p))
    json.dumps(boot)  # serializable as written to bootstrap_comparison.json


def test_bootstrap_invalid_inputs(scored):
    y, preds = scored
    with pytest.raises(KeyError):
        cmp.paired_bootstrap(y, preds, n_replicates=5, reference="nope")
    with pytest.raises(ValueError):
        cmp.bootstrap_replicates(np.zeros_like(y), preds, n_replicates=5)
    with pytest.raises(ValueError):
        cmp.bootstrap_replicates(y, preds, n_replicates=0)
    with pytest.raises(ValueError):
        cmp.bootstrap_replicates(y, {}, n_replicates=5)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_comparison_figures_are_written(scored, tmp_path):
    y, preds = scored
    boot = cmp.paired_bootstrap(y, preds, n_replicates=10, seed=1)
    paths = cmp.make_comparison_figures(y, preds, boot, figures_dir=tmp_path, prefix="unit")
    assert set(paths) == {
        "roc_comparison",
        "precision_recall_comparison",
        "calibration_comparison",
        "probability_distributions",
        "delta_bootstrap",
    }
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 1000, path
    assert cmp.make_comparison_figures(
        y, preds, None, figures_dir=tmp_path, prefix="nb"
    ).keys() >= {"roc_comparison"}
