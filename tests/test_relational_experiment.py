"""Relational experiment: configurations, elimination, importance, ablation smoke run."""

import json

import numpy as np
import pandas as pd
import pytest

from riskpilot import config
from riskpilot.data.load import split_features_target
from riskpilot.features.relational import assemble
from riskpilot.models import challengers as ch
from riskpilot.models import relational_experiment as rx

ID = config.ID_COL


# --------------------------------------------------------------------------- #
# Configurations and naming
# --------------------------------------------------------------------------- #
def test_ablation_configs_cover_sequence_single_and_leave_one_out():
    configs = rx.ablation_configs(("a", "b", "c"))
    names = [name for name, _, _ in configs]
    # leaving out the last source is the cumulative M2 model: not fitted twice
    assert names == ["M0_application", "M1_+a", "M2_+b", "M3_+c", "S_b", "S_c", "L-a", "L-b"]
    by_name = {name: sources for name, _, sources in configs}
    assert by_name["M3_+c"] == ("a", "b", "c") and by_name["S_c"] == ("c",)
    assert by_name["L-b"] == ("a", "c") and by_name["M0_application"] == ()
    groups = {group for _, group, _ in configs}
    assert groups == {"sequential", "single", "leave-one-out"}
    assert (
        len(rx.ablation_configs(("a", "b"), include_single=False, include_leave_one_out=False)) == 3
    )


def test_config_names_are_consistent_with_ablation_configs():
    sources = ("a", "b", "c")
    for name, _, subset in rx.ablation_configs(sources):
        assert rx._config_name(subset, sources) == name
    assert rx._config_name(("a",), sources) == "M1_+a"
    assert rx._config_name(("b",), sources) == "S_b"
    assert rx._config_name(("a", "b"), sources) == "M2_+b"
    assert rx._config_name(("b", "c"), sources) == "L-a"
    assert rx._config_name(("c",), ("a", "b", "c", "d")) == "S_c"
    assert rx._config_name(("a", "d"), ("a", "b", "c", "d")) == "L-b-c"


def test_columns_for_sources_keeps_application_and_requested_prefixes():
    columns = [
        "AMT_CREDIT",
        "bureau__credit_count",
        "bureau__has_history",
        "pos__dpd_max",
        "previous__count",
    ]
    assert rx.columns_for_sources(columns, ()) == ["AMT_CREDIT"]
    assert rx.columns_for_sources(columns, ("bureau",)) == [
        "AMT_CREDIT",
        "bureau__credit_count",
        "bureau__has_history",
    ]
    assert rx.columns_for_sources(columns, ("pos", "previous")) == [
        "AMT_CREDIT",
        "pos__dpd_max",
        "previous__count",
    ]


def test_locked_spec_reads_the_milestone_2_selection(tmp_path):
    payload = {
        "libraries": {
            "lightgbm": {
                "locked_spec": {
                    "library": "lightgbm",
                    "params": {
                        **ch.DEFAULT_PARAMS["lightgbm"],
                        "n_estimators": 1380,
                        "num_leaves": 15,
                    },
                    "heavy_tail": "none",
                    "missing_indicators": "none",
                }
            }
        }
    }
    path = tmp_path / "challenger_selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    spec = rx.locked_lightgbm_spec(path, n_jobs=2)
    assert spec.params["n_estimators"] == 1380 and spec.params["num_leaves"] == 15
    assert spec.params["n_jobs"] == 2 and spec.library == "lightgbm"
    with pytest.raises(FileNotFoundError):
        rx.locked_lightgbm_spec(tmp_path / "missing.json")


# --------------------------------------------------------------------------- #
# Backward elimination on synthetic validation predictions
# --------------------------------------------------------------------------- #
def _synthetic_predictions(strength: dict[str, float], n: int = 3000, seed: int = 0):
    """Validation outcome + predictions per configuration (a source helps when included)."""
    rng = np.random.default_rng(seed)
    signals = {s: rng.normal(size=n) for s in strength}
    logit = -2.5 + sum(w * signals[s] for s, w in strength.items())
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

    def predict(sources):
        included = -2.5 + sum(strength[s] * signals[s] for s in sources)
        return 1 / (1 + np.exp(-(included + rng.normal(scale=0.05, size=n))))

    return y, predict


def test_backward_elimination_drops_only_useless_sources():
    sources = ("strong", "useless", "weak")
    strength = {"strong": 1.5, "useless": 0.0, "weak": 0.8}  # weak but resolvable at n=3000
    y, predict = _synthetic_predictions(strength)
    predictions, records = {}, {}
    for name, _, subset in rx.ablation_configs(sources):
        predictions[name] = predict(subset)
        records[name] = {"log_loss": 0.0}
    calls = []

    def evaluate_missing(name, subset):
        calls.append(name)
        return {"log_loss": 0.0}, predict(subset)

    out = rx.backward_elimination(
        y, predictions, records, sources, evaluate_missing=evaluate_missing, n_bootstrap=100, seed=1
    )
    assert out["retained"] == ["strong", "weak"]
    assert out["final_config"] == "L-useless"
    assert out["rounds"][0]["weakest"] == "useless" and out["rounds"][0]["dropped"] is True
    assert (
        out["rounds"][-1]["dropped"] is False
    )  # stopped: removing 'weak' or 'strong' reliably hurts
    assert "L-useless" in predictions and all(name in predictions for name in calls)


def test_backward_elimination_keeps_everything_when_all_sources_matter():
    sources = ("a", "b")
    y, predict = _synthetic_predictions({"a": 1.2, "b": 1.0})
    predictions = {name: predict(subset) for name, _, subset in rx.ablation_configs(sources)}
    records = {name: {"log_loss": 0.0} for name in predictions}
    out = rx.backward_elimination(
        y,
        predictions,
        records,
        sources,
        evaluate_missing=lambda n, s: (None, None),
        n_bootstrap=100,
        seed=2,
    )
    assert out["retained"] == ["a", "b"] and out["final_config"] == "M2_+b"
    assert len(out["rounds"]) == 1 and out["rounds"][0]["dropped"] is False


# --------------------------------------------------------------------------- #
# Importance by source and figures
# --------------------------------------------------------------------------- #
def test_importance_by_source_groups_features(synthetic_df, tmp_path):
    X, y = split_features_target(synthetic_df)
    X = X.assign(bureau__credit_count=np.arange(len(X), dtype="float32") % 5, pos__dpd_max=0.0)
    spec = ch.ChallengerSpec.default(
        "lightgbm", n_estimators=15, min_child_samples=5, num_leaves=7, n_jobs=2
    )
    result = ch.fit_challenger(spec, X, y)
    table = rx.importance_by_source(result.model, result.preprocessor.get_feature_names_out())
    assert list(table["source"]) == ["application", "bureau", "pos"]
    assert table.set_index("source").loc["bureau", "n_features"] == 1
    assert table["gain_share"].sum() == pytest.approx(1.0)
    assert (table["features_used"] <= table["n_features"]).all()
    fig = rx.plot_importance_by_source(table, path=tmp_path / "imp.png")
    assert (tmp_path / "imp.png").stat().st_size > 1000
    ablation = pd.DataFrame(
        {
            "configuration": ["M0_application", "M1_+bureau", "S_pos", "L-bureau"],
            "group": ["sequential", "sequential", "single", "leave-one-out"],
            "n_features": [9, 10, 10, 10],
            "roc_auc": [0.70, 0.72, 0.71, 0.705],
            "log_loss": [0.30, 0.29, 0.295, 0.298],
        }
    )
    rx.plot_ablation(ablation, path=tmp_path / "abl.png")
    assert (tmp_path / "abl.png").stat().st_size > 1000
    del fig


# --------------------------------------------------------------------------- #
# Ablation + retune stages on synthetic data (tiny models, tiny tables)
# --------------------------------------------------------------------------- #
@pytest.fixture
def relational_setup(synthetic_df, tmp_path):
    """Tiny relational tables for the synthetic customers, plus a Milestone 2 selection file."""
    ids = synthetic_df[ID].to_numpy()
    rng = np.random.default_rng(5)
    processed = tmp_path / "relational"
    processed.mkdir()
    tables = {}
    for source in assemble.SOURCES:
        module = assemble.SOURCE_MODULES[source]
        chosen = rng.choice(ids, size=len(ids) // 2, replace=False)
        frame = pd.DataFrame({ID: chosen})
        for spec in module.SPECS:
            frame[f"{module.PREFIX}__{spec.name}"] = rng.normal(size=len(chosen)).astype("float32")
        frame.to_parquet(processed / f"{source}.parquet", index=False)
        tables[source] = frame
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    selection = {
        "libraries": {
            "lightgbm": {
                "locked_spec": ch.ChallengerSpec.default(
                    "lightgbm", n_estimators=30, num_leaves=7, min_child_samples=5, n_jobs=2
                ).to_dict()
            }
        }
    }
    (metrics / "challenger_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    paths = rx.RelationalPaths(
        processed_dir=tmp_path / "processed",
        relational_dir=processed,
        metrics_dir=metrics,
        figures_dir=tmp_path / "figures",
        models_dir=tmp_path / "models",
    )
    return paths, tables


def test_ablation_and_retune_stages_smoke(synthetic_df, relational_setup):
    paths, _ = relational_setup
    X, y = split_features_target(synthetic_df)
    ids = synthetic_df[ID]
    cfg = rx.AblationConfig(
        n_jobs=2,
        n_bootstrap_validation=20,
        sources=("bureau", "pos"),
        param_overrides={"n_estimators": 20, "min_child_samples": 5, "num_leaves": 7},
        retune_axes=[("num_leaves", [3, 7])],
        early_stopping_rounds=5,
    )
    selection = rx.run_ablation_stage(X, y, ids, paths, cfg)

    ablation = pd.read_csv(paths.ablation_path)
    expected = {name for name, _, _ in rx.ablation_configs(("bureau", "pos"))}
    assert expected <= set(ablation["configuration"])
    assert {
        "n_features",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier",
        "brier_skill",
        "ece",
        "fit_seconds",
    } <= set(ablation.columns)
    m0 = ablation.set_index("configuration").loc["M0_application"]
    assert m0["n_features"] == X.shape[1]
    full = ablation.set_index("configuration").loc["M2_+pos"]
    n_rel = sum(len(assemble.SOURCE_MODULES[s].SPECS) + 1 for s in ("bureau", "pos"))
    assert full["n_features"] == X.shape[1] + n_rel
    membership = pd.read_csv(paths.internal_membership_path)
    assert set(membership["internal_split"]) == {"fit", "validation"} and len(membership) == len(X)
    assert set(selection["retained_sources"]) <= {"bureau", "pos"}
    assert (
        selection["elimination"]["rounds"] and "candidates" in selection["elimination"]["rounds"][0]
    )
    preds = pd.read_parquet(paths.validation_predictions_path)
    assert len(preds) == selection["validation_split"]["n_valid"]

    # Resume: nothing is refitted for configurations already recorded.
    again = rx.run_ablation_stage(X, y, ids, paths, cfg, resume=True)
    assert again["retained_sources"] == selection["retained_sources"]

    retune = rx.run_retune_stage(X, y, ids, selection, paths, cfg)
    trials = pd.read_csv(paths.trials_path)
    assert len(trials) == retune["n_trials"] >= 2
    assert retune["locked_spec"]["params"]["n_estimators"] >= 1
    assert "changed_vs_frozen" in retune
    saved = json.loads(paths.selection_path.read_text(encoding="utf-8"))
    assert "retune" in saved and saved["retune"]["selected_trial"] == retune["selected_trial"]


def test_ablation_stage_never_receives_test_data():
    import inspect

    params = list(inspect.signature(rx.run_ablation_stage).parameters)
    assert params[:3] == ["X_train", "y_train", "ids_train"]
    assert not any("test" in p.lower() for p in params)
    params = list(inspect.signature(rx.run_retune_stage).parameters)
    assert not any("test" in p.lower() for p in params)
