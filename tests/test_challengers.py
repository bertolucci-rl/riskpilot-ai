import inspect
import json
import subprocess
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.pipeline import Pipeline

from riskpilot.data.load import split_features_target
from riskpilot.models import challengers as ch
from riskpilot.models.train import BaselineConfig, make_split, run_baseline

QUICK_OVERRIDES = {
    "lightgbm": {**ch.SMOKE_PARAM_OVERRIDES["lightgbm"], "n_jobs": 2},
    "xgboost": {**ch.SMOKE_PARAM_OVERRIDES["xgboost"], "n_jobs": 2},
}


def quick_spec(library: str, **overrides) -> ch.ChallengerSpec:
    return ch.ChallengerSpec.default(library, **QUICK_OVERRIDES[library], **overrides)


@pytest.fixture
def data(synthetic_df):
    X, y = split_features_target(synthetic_df)
    X_fit, X_valid, y_fit, y_valid = make_split(X, y, test_size=0.25, random_state=42)
    return X_fit, X_valid, y_fit, y_valid


# --------------------------------------------------------------------------- #
# Specification and construction
# --------------------------------------------------------------------------- #
def test_default_specs_are_deterministic_and_unweighted():
    for library in ch.LIBRARIES:
        spec = ch.ChallengerSpec.default(library)
        assert spec.params["random_state"] == 42
        assert "n_jobs" in spec.params
        assert "class_weight" not in spec.params
        assert "scale_pos_weight" not in spec.params
        assert spec.heavy_tail == "none" and spec.missing_indicators == "none"
    with pytest.raises(ValueError):
        ch.ChallengerSpec(library="catboost", params={})


def test_spec_is_immutable_and_round_trips():
    spec = ch.ChallengerSpec.default("lightgbm")
    changed = spec.with_params(num_leaves=7).with_preprocessing(heavy_tail="log1p")
    assert spec.params["num_leaves"] == 31 and spec.heavy_tail == "none"
    assert changed.params["num_leaves"] == 7 and changed.heavy_tail == "log1p"
    assert ch.ChallengerSpec.from_dict(json.loads(json.dumps(changed.to_dict()))) == changed


def test_build_challenger_pipeline_structure():
    lgbm = ch.build_challenger(quick_spec("lightgbm"))
    xgbm = ch.build_challenger(quick_spec("xgboost"), early_stopping_rounds=7)
    assert isinstance(lgbm, Pipeline) and [n for n, _ in lgbm.steps] == ["preprocess", "model"]
    assert isinstance(lgbm["model"], lgb.LGBMClassifier)
    assert isinstance(xgbm["model"], xgb.XGBClassifier)
    assert xgbm["model"].get_params()["early_stopping_rounds"] == 7
    assert ch.build_estimator(quick_spec("xgboost")).get_params()["early_stopping_rounds"] is None


# --------------------------------------------------------------------------- #
# Fitting, early stopping, locking, persistence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("library", ch.LIBRARIES)
def test_early_stopping_lock_and_refit_reproduce_predictions(library, data, tmp_path):
    X_fit, X_valid, y_fit, y_valid = data
    result = ch.fit_challenger(
        quick_spec(library), X_fit, y_fit, X_valid=X_valid, y_valid=y_valid, early_stopping_rounds=5
    )
    assert isinstance(result.best_iteration, int) and result.n_trees >= 1
    assert result.n_features == X_fit.shape[1]
    p_valid = result.pipeline.predict_proba(X_valid)[:, 1]
    assert p_valid.shape == (len(X_valid),) and np.all((p_valid >= 0) & (p_valid <= 1))

    locked = ch.lock_spec(result)
    assert locked.params["n_estimators"] == result.n_trees
    refit = ch.fit_challenger(locked, X_fit, y_fit)  # no validation set -> no early stopping
    assert refit.best_iteration is None and refit.n_trees == result.n_trees
    np.testing.assert_allclose(refit.pipeline.predict_proba(X_valid)[:, 1], p_valid, atol=1e-6)

    # Deterministic under a fixed seed, and identical after a joblib round trip.
    again = ch.fit_challenger(locked, X_fit, y_fit)
    np.testing.assert_array_equal(
        again.pipeline.predict_proba(X_valid)[:, 1], refit.pipeline.predict_proba(X_valid)[:, 1]
    )
    path = tmp_path / f"{library}.joblib"
    joblib.dump(refit.pipeline, path)
    reloaded = joblib.load(path)
    np.testing.assert_array_equal(
        reloaded.predict_proba(X_valid)[:, 1], refit.pipeline.predict_proba(X_valid)[:, 1]
    )


@pytest.mark.parametrize("library", ch.LIBRARIES)
def test_fitted_challenger_handles_unseen_rows(library, data):
    X_fit, X_valid, y_fit, y_valid = data
    result = ch.fit_challenger(quick_spec(library).with_params(n_estimators=10), X_fit, y_fit)
    row = X_valid.head(1).copy()
    row["OCCUPATION_TYPE"] = "Astronaut"
    row["EXT_SOURCE_1"] = np.nan
    row["DAYS_EMPLOYED"] = 365243.0
    p = result.pipeline.predict_proba(row)[:, 1]
    assert 0.0 <= p[0] <= 1.0


def test_fit_requires_y_valid_with_x_valid(data):
    X_fit, X_valid, y_fit, _ = data
    with pytest.raises(ValueError, match="y_valid"):
        ch.fit_challenger(quick_spec("lightgbm"), X_fit, y_fit, X_valid=X_valid)


# --------------------------------------------------------------------------- #
# Search, ablations and the selection stage (validation data only)
# --------------------------------------------------------------------------- #
def test_coordinate_search_records_every_trial(data, tmp_path):
    X_fit, X_valid, y_fit, y_valid = data
    log = ch.TrialLog(tmp_path / "trials.csv")
    best_spec, best = ch.coordinate_search(
        quick_spec("lightgbm"),
        [("num_leaves", [7, 15, 31])],  # 31 is the starting value: not refitted
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        log=log,
        early_stopping_rounds=5,
    )
    frame = pd.read_csv(tmp_path / "trials.csv")
    assert len(frame) == 3 and list(frame["stage"]) == [
        "initial",
        "search:num_leaves",
        "search:num_leaves",
    ]
    assert best["val_log_loss"] == frame["val_log_loss"].min()
    assert best_spec.params["num_leaves"] == json.loads(best["params_json"])["num_leaves"]
    for col in (
        "val_roc_auc",
        "val_average_precision",
        "val_brier_score",
        "val_expected_calibration_error",
        "n_trees",
        "fit_seconds",
    ):
        assert col in frame.columns and frame[col].notna().all()


def test_selection_stage_never_receives_test_data():
    params = inspect.signature(ch.run_selection_stage).parameters
    assert list(params) == ["X_train", "y_train", "cfg", "log"]
    assert not any("test" in name.lower() for name in params)


def test_run_selection_stage_smoke(synthetic_df, tmp_path):
    X, y = split_features_target(synthetic_df)
    cfg = ch.SelectionConfig(
        libraries=("lightgbm",),
        search_axes={"lightgbm": [("num_leaves", [7, 15])]},
        param_overrides=QUICK_OVERRIDES,
        early_stopping_rounds=5,
        heavy_tail_treatments=("log1p",),
        indicator_variants=("deduplicated",),
        lr_max_iter=300,
    )
    log = ch.TrialLog(tmp_path / "trials.csv")
    selection = ch.run_selection_stage(X, y, cfg, log=log)

    split = selection["validation_split"]
    assert split["n_fit"] + split["n_valid"] == len(y)
    assert "logistic_regression" in selection["reference"]
    entry = selection["libraries"]["lightgbm"]
    assert entry["locked_spec"]["params"]["n_estimators"] == entry["locked_n_estimators"]
    assert set(entry["ablations"]) == {"heavy_tail", "missing_indicators"}
    for ablation in entry["ablations"].values():
        assert "adopted" in ablation and "gain_in_val_log_loss" in ablation
    # 1 initial + 2 search values + 1 heavy-tail + 1 indicator trial, plus the LR reference row.
    assert entry["n_trials"] == 5 and len(log.records) == 6
    json.dumps(selection)  # machine-readable


# --------------------------------------------------------------------------- #
# Command-line entry point (baseline first, then challengers, tiny models)
# --------------------------------------------------------------------------- #
def test_cli_quick_end_to_end(synthetic_csv, tmp_path, capsys):
    dirs = {
        "figures_dir": tmp_path / "figures",
        "metrics_dir": tmp_path / "metrics",
        "models_dir": tmp_path / "models",
        "processed_dir": tmp_path / "processed",
    }
    run_baseline(BaselineConfig(data_path=synthetic_csv, max_iter=500, **dirs))
    exit_code = ch.main(
        [
            "--data-path",
            str(synthetic_csv),
            "--quick",
            "--n-jobs",
            "2",
            "--n-bootstrap",
            "20",
            *[
                arg
                for key, value in dirs.items()
                for arg in (f"--{key.replace('_', '-')}", str(value))
            ],
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Frozen test split comparison" in out and "XGBoost" in out

    metrics_dir = dirs["metrics_dir"]
    for name in (
        "challenger_trials.csv",
        "challenger_selection.json",
        "lightgbm_metrics.json",
        "xgboost_metrics.json",
        "model_comparison.csv",
        "bootstrap_comparison.json",
    ):
        assert (metrics_dir / name).is_file(), name
    for lib in ch.LIBRARIES:
        payload = json.loads((metrics_dir / f"{lib}_metrics.json").read_text(encoding="utf-8"))
        assert payload["model"]["class_weight"] is None
        assert "scale_pos_weight" not in payload["spec"]["params"]
        assert payload["model"]["early_stopping"].startswith("none")
        assert payload["split"]["verified_against_reseeded_split"] is True
        assert (dirs["models_dir"] / f"{lib}_challenger.joblib").is_file()
    table = pd.read_csv(metrics_dir / "model_comparison.csv")
    assert list(table["key"]) == ["logistic_regression", "lightgbm", "xgboost"]
    assert table.loc[0, "delta_roc_auc"] == 0.0
    boot = json.loads((metrics_dir / "bootstrap_comparison.json").read_text(encoding="utf-8"))
    assert boot["n_replicates"] == 20 and "lightgbm_vs_logistic_regression" in boot["differences"]
    for name in ("challenger_roc_comparison.png", "challenger_delta_bootstrap.png"):
        assert (dirs["figures_dir"] / name).stat().st_size > 1000
    assert (dirs["processed_dir"] / "challenger_test_predictions.csv").is_file()


def test_cli_missing_membership_returns_error_code(synthetic_csv, tmp_path):
    exit_code = ch.main(
        [
            "--data-path",
            str(synthetic_csv),
            "--quick",
            "--processed-dir",
            str(tmp_path / "nope"),
            "--metrics-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 2


def test_module_entry_point_does_not_double_import():
    proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "riskpilot.models.challengers",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "found in sys.modules" not in proc.stderr
    assert "usage:" in proc.stdout


def test_resumed_selection_reuses_recorded_trials_without_refitting(synthetic_df, tmp_path):
    X, y = split_features_target(synthetic_df)
    cfg = ch.SelectionConfig(
        libraries=("lightgbm",),
        search_axes={"lightgbm": [("num_leaves", [7, 15])]},
        param_overrides=QUICK_OVERRIDES,
        early_stopping_rounds=5,
        heavy_tail_treatments=("log1p",),
        indicator_variants=("deduplicated",),
        lr_max_iter=300,
    )
    first = ch.run_selection_stage(X, y, cfg, log=ch.TrialLog(tmp_path / "trials.csv"))
    recorded = pd.read_csv(tmp_path / "trials.csv")

    resumed_log = ch.TrialLog(tmp_path / "trials.csv", resume=True)
    assert len(resumed_log.records) == len(recorded)
    second = ch.run_selection_stage(X, y, cfg, log=resumed_log)
    assert second["n_trials_reused_from_log"] == len(recorded)  # nothing refitted, LR included
    assert len(pd.read_csv(tmp_path / "trials.csv")) == len(recorded)  # no duplicates written
    assert (
        second["libraries"]["lightgbm"]["locked_spec"]
        == first["libraries"]["lightgbm"]["locked_spec"]
    )
    assert second["libraries"]["lightgbm"]["validation_metrics"] == pytest.approx(
        first["libraries"]["lightgbm"]["validation_metrics"]
    )
    assert second["reference"]["logistic_regression"]["validation_metrics"][
        "log_loss"
    ] == pytest.approx(first["reference"]["logistic_regression"]["validation_metrics"]["log_loss"])
