import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from riskpilot import config
from riskpilot.data.load import split_features_target
from riskpilot.models.train import BaselineConfig, _display_path, main, make_split, run_baseline


def test_make_split_is_stratified_and_disjoint(synthetic_df):
    X, y = split_features_target(synthetic_df)
    X_train, X_test, y_train, y_test = make_split(X, y, test_size=0.25, random_state=42)
    assert len(X_train) + len(X_test) == len(X)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.03
    # Deterministic under a fixed seed.
    _, X_test_again, _, _ = make_split(X, y, test_size=0.25, random_state=42)
    assert list(X_test.index) == list(X_test_again.index)


def test_display_path_is_relative_inside_the_project(tmp_path):
    assert _display_path(config.FIGURES_DIR / "x.png") == "artifacts/figures/x.png"
    outside = tmp_path / "y.png"
    assert _display_path(outside) == outside.as_posix()


def _config(synthetic_csv: Path, tmp_path: Path, **overrides) -> BaselineConfig:
    base = {
        "data_path": synthetic_csv,
        "figures_dir": tmp_path / "figures",
        "metrics_dir": tmp_path / "metrics",
        "models_dir": tmp_path / "models",
        "processed_dir": tmp_path / "processed",
        "max_iter": 500,
        "prefix": "unit",
    }
    base.update(overrides)
    return BaselineConfig(**base)


def test_run_baseline_end_to_end(synthetic_csv, tmp_path):
    result = run_baseline(_config(synthetic_csv, tmp_path))

    payload = result.payload
    assert payload["model"]["converged"] is True
    # Recorded by name, not via the attribute scikit-learn 1.8 turned into "deprecated".
    assert payload["model"]["penalty"] == "l2"
    assert payload["split"]["n_train"] + payload["split"]["n_test"] == payload["data"]["n_rows"]
    test_metrics = payload["metrics"]["test"]
    assert 0.0 <= test_metrics["roc_auc"] <= 1.0
    assert test_metrics["n"] == payload["split"]["n_test"]
    assert result.y_prob_test.shape == (payload["split"]["n_test"],)

    # Artifacts exist and the JSON round-trips.
    assert result.metrics_path.is_file()
    reloaded = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert reloaded["metrics"]["test"]["roc_auc"] == pytest.approx(test_metrics["roc_auc"])
    for fig in payload["artifacts"]["figures"].values():
        assert Path(fig).is_file()
    assert result.model_path is not None and result.model_path.is_file()

    # The persisted pipeline reproduces the in-memory predictions.
    pipe = joblib.load(result.model_path)
    X, _ = split_features_target(pd.read_csv(synthetic_csv))
    membership = pd.read_csv(tmp_path / "processed" / "unit_split_membership.csv")
    assert membership["split"].value_counts()["test"] == payload["split"]["n_test"]
    test_ids = membership.loc[membership["split"] == "test", config.ID_COL]
    df = pd.read_csv(synthetic_csv)
    X_test = df[df[config.ID_COL].isin(test_ids)].pipe(split_features_target)[0]
    reproduced = pipe.predict_proba(X_test)[:, 1]
    assert reproduced.shape == result.y_prob_test.shape
    assert np.allclose(np.sort(reproduced), np.sort(result.y_prob_test))
    assert X.shape[1] == payload["data"]["n_raw_features"]


def test_run_baseline_without_persistence(synthetic_csv, tmp_path):
    result = run_baseline(_config(synthetic_csv, tmp_path, save_model=False, save_split=False))
    assert result.model_path is None
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "processed").exists()
    assert result.metrics_path.is_file()


def test_cli_smoke(synthetic_csv, tmp_path, capsys):
    exit_code = main(
        [
            "--data-path",
            str(synthetic_csv),
            "--nrows",
            "300",
            "--max-iter",
            "500",
            "--prefix",
            "cli",
            "--no-save-model",
            "--figures-dir",
            str(tmp_path / "f"),
            "--metrics-dir",
            str(tmp_path / "m"),
            "--processed-dir",
            str(tmp_path / "p"),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "m" / "cli_metrics.json").is_file()
    out = capsys.readouterr().out
    assert "ROC-AUC" in out and "Brier score" in out


def test_cli_missing_data_returns_error_code(tmp_path):
    exit_code = main(["--data-path", str(tmp_path / "missing.csv"), "--metrics-dir", str(tmp_path)])
    assert exit_code == 2


def test_module_entry_point_does_not_double_import():
    """``python -m riskpilot.models.train`` must not trigger runpy's double-import warning.

    That warning appears when the package ``__init__`` imports the module being
    executed with ``-m``; with ``-W error`` it would abort the run.
    """
    proc = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-m", "riskpilot.models.train", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "found in sys.modules" not in proc.stderr
    assert "usage:" in proc.stdout
