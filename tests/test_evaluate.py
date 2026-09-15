import json

import numpy as np
import pytest

from riskpilot.models import evaluate as ev


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = y.astype(float) * 0.98 + 0.01
    m = ev.compute_metrics(y, p, n_bins=2)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["average_precision"] == pytest.approx(1.0)
    assert m["brier_score"] < 0.001
    assert m["log_loss"] < 0.02


def test_constant_prediction_matches_reference_baselines():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=2000) < 0.08).astype(int)
    prevalence = y.mean()
    p = np.full(y.shape, prevalence)
    m = ev.compute_metrics(y, p)
    assert m["brier_score"] == pytest.approx(prevalence * (1 - prevalence))
    assert m["brier_score"] == pytest.approx(m["brier_prevalence_baseline"])
    assert m["brier_skill_score"] == pytest.approx(0.0)
    assert m["log_loss"] == pytest.approx(m["log_loss_prevalence_baseline"])
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["expected_calibration_error"] == pytest.approx(0.0, abs=1e-12)


def test_metrics_are_json_serializable():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 0])
    p = np.array([0.1, 0.8, 0.3, 0.6, 0.9, 0.2, 0.05, 0.4])
    json.dumps(ev.compute_metrics(y, p, n_bins=4))


@pytest.mark.parametrize(
    "y, p",
    [
        ([0, 1], [0.5]),  # shape mismatch
        ([0, 1], [0.5, 1.2]),  # out of range
        ([0, 1], [0.5, np.nan]),  # nan
        ([0, 2], [0.5, 0.6]),  # non-binary target
        ([], []),  # empty
    ],
)
def test_invalid_inputs_raise(y, p):
    with pytest.raises(ValueError):
        ev.compute_metrics(np.array(y), np.array(p))


def test_calibration_table_partitions_all_rows():
    rng = np.random.default_rng(2)
    p = rng.beta(1, 9, size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    table = ev.calibration_table(y, p, n_bins=10)
    assert table["n"].sum() == 1000
    assert (table["mean_predicted"].diff().dropna() > 0).all()  # bins increase monotonically
    assert set(table.columns) >= {"bin", "lower", "upper", "n", "mean_predicted", "observed_rate"}
    ece = ev.expected_calibration_error(table)
    assert 0.0 <= ece < 0.1  # simulated data is calibrated by construction


def test_calibration_table_uniform_strategy_and_bad_strategy():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    table = ev.calibration_table(y, p, n_bins=4, strategy="uniform")
    assert table["n"].sum() == 4
    with pytest.raises(ValueError):
        ev.calibration_table(y, p, strategy="nope")


def test_figures_are_written(tmp_path):
    rng = np.random.default_rng(3)
    p = rng.beta(1, 9, size=500)
    y = (rng.uniform(size=500) < p).astype(int)
    paths = ev.make_evaluation_figures(y, p, figures_dir=tmp_path, prefix="unit")
    assert set(paths) == {
        "roc_curve",
        "precision_recall_curve",
        "calibration_curve",
        "probability_distribution",
    }
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 1000, path


def test_save_metrics_creates_parents(tmp_path):
    out = ev.save_metrics({"a": np.float64(1.5), "b": [np.int64(1)]}, tmp_path / "x" / "m.json")
    assert json.loads(out.read_text()) == {"a": 1.5, "b": [1]}
