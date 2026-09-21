"""Relational feature builders, assembly safety, catalog and cached build (tiny fixtures)."""

import json

import numpy as np
import pandas as pd
import pytest

from riskpilot import config
from riskpilot.features.relational import (
    assemble,
    bureau,
    credit_card,
    installments,
    pos_cash,
    previous,
)
from riskpilot.features.relational import build as build_mod
from riskpilot.features.relational.common import FeatureSpec, finalize_customer_table

ID = config.ID_COL


# --------------------------------------------------------------------------- #
# Fixtures: three customers (100, 200, 300); 300 never has history
# --------------------------------------------------------------------------- #
@pytest.fixture
def bureau_frames():
    b = pd.DataFrame(
        {
            "SK_ID_CURR": [100, 100, 200],
            "SK_ID_BUREAU": [1, 2, 3],
            "CREDIT_ACTIVE": pd.Categorical(["Active", "Closed", "Sold"]),
            "CREDIT_TYPE": pd.Categorical(["Consumer credit", "Credit card", "Mortgage"]),
            "DAYS_CREDIT": [-100.0, -800.0, -30.0],
            "CREDIT_DAY_OVERDUE": [5.0, 0.0, 0.0],
            "DAYS_CREDIT_ENDDATE": [300.0, -400.0, 1000.0],
            "DAYS_ENDDATE_FACT": [np.nan, -350.0, np.nan],
            "AMT_CREDIT_MAX_OVERDUE": [50.0, np.nan, 0.0],
            "CNT_CREDIT_PROLONG": [0.0, 1.0, 0.0],
            "AMT_CREDIT_SUM": [1000.0, 500.0, 2000.0],
            "AMT_CREDIT_SUM_DEBT": [400.0, -20.0, 1500.0],
            "AMT_CREDIT_SUM_LIMIT": [0.0, 100.0, -5.0],
            "AMT_CREDIT_SUM_OVERDUE": [10.0, 0.0, 0.0],
            "DAYS_CREDIT_UPDATE": [-10.0, 20.0, -5.0],
            "AMT_ANNUITY": [50.0, np.nan, 100.0],
        }
    )
    bb = pd.DataFrame(
        {
            "SK_ID_BUREAU": [1, 1, 1, 1, 3, 3],
            "MONTHS_BALANCE": [0, -1, -2, -3, 0, -20],
            "STATUS": pd.Categorical(["0", "1", "C", "X", "3", "0"]),
        }
    )
    return b, bb


def test_bureau_balance_aggregation(bureau_frames):
    _, bb = bureau_frames
    s = bureau.aggregate_bureau_balance(bb).set_index("SK_ID_BUREAU")
    assert s.loc[1, "bb_months"] == 4
    assert s.loc[1, "bb_known_months"] == 3  # X is unknown
    assert s.loc[1, "bb_dpd_months"] == 1
    assert s.loc[1, "bb_status_max"] == 1
    assert s.loc[1, "bb_dpd_share"] == pytest.approx(1 / 3)
    assert s.loc[1, "bb_closed_share"] == pytest.approx(0.25)
    assert s.loc[1, "bb_history_len"] == 4
    assert not s.loc[1, "bb_latest_dpd"]  # latest month (0) has status 0
    assert s.loc[3, "bb_latest_dpd"] and s.loc[3, "bb_severe"] and s.loc[3, "bb_recent_dpd"]


def test_bureau_features(bureau_frames):
    b, bb = bureau_frames
    out = bureau.build_bureau_features(b, bb).set_index(ID)
    assert list(out.index) == [100, 200]  # customer 300 has no rows: absent, not zero-filled here
    assert out.loc[100, "bureau__credit_count"] == 2
    assert out.loc[100, "bureau__active_ratio"] == pytest.approx(0.5)
    assert (
        out.loc[100, "bureau__consumer_count"] == 1
        and out.loc[100, "bureau__credit_card_count"] == 1
    )
    assert out.loc[100, "bureau__debt_sum_total"] == pytest.approx(400.0)  # -20 clipped to 0
    assert out.loc[100, "bureau__limit_sum"] == pytest.approx(100.0)
    assert out.loc[100, "bureau__debt_to_credit_ratio"] == pytest.approx(400 / 1500)
    assert out.loc[100, "bureau__overdue_to_debt_ratio"] == pytest.approx(10 / 400)
    assert out.loc[100, "bureau__days_update_max"] == 0.0  # +20 clipped to 0
    assert out.loc[100, "bureau__enddate_remaining_max"] == 300.0  # only the active credit
    assert out.loc[100, "bureau__credits_last_year"] == 1
    assert out.loc[100, "bureau__bb_credits_with_history"] == 1
    assert out.loc[100, "bureau__bb_dpd_months_sum"] == 1
    assert out.loc[200, "bureau__sold_or_bad_count"] == 1
    assert out.loc[200, "bureau__bb_severe_credits"] == 1
    assert out.loc[200, "bureau__bb_latest_dpd_credits"] == 1
    assert out.loc[200, "bureau__debt_to_credit_ratio"] == pytest.approx(0.75)
    assert set(out.columns) == {f"bureau__{s.name}" for s in bureau.SPECS}
    assert all(out[c].dtype == np.float32 for c in out.columns)


def test_bureau_without_balance(bureau_frames):
    b, _ = bureau_frames
    out = bureau.build_bureau_features(b, None).set_index(ID)
    assert out.loc[100, "bureau__bb_credits_with_history"] == 0
    assert np.isnan(out.loc[100, "bureau__bb_status_max"])


@pytest.fixture
def previous_frame():
    return pd.DataFrame(
        {
            "SK_ID_PREV": [1, 2, 3, 4],
            "SK_ID_CURR": [100, 100, 100, 200],
            "NAME_CONTRACT_TYPE": pd.Categorical(
                ["Cash loans", "Consumer loans", "Cash loans", "Revolving loans"]
            ),
            "AMT_ANNUITY": [10.0, 20.0, np.nan, 5.0],
            "AMT_APPLICATION": [1000.0, 0.0, 3000.0, 500.0],
            "AMT_CREDIT": [1100.0, 0.0, 0.0, 600.0],
            "AMT_DOWN_PAYMENT": [0.0, np.nan, 100.0, 0.0],
            "AMT_GOODS_PRICE": [1000.0, np.nan, 3000.0, 500.0],
            "RATE_DOWN_PAYMENT": [0.0, np.nan, 0.1, 0.0],
            "NAME_CONTRACT_STATUS": pd.Categorical(["Approved", "Canceled", "Refused", "Approved"]),
            "DAYS_DECISION": [-500.0, -100.0, -30.0, -900.0],
            "CODE_REJECT_REASON": pd.Categorical(["XAP", "XAP", "SCO", "XAP"]),
            "NAME_CLIENT_TYPE": pd.Categorical(["New", "Repeater", "Repeater", "New"]),
            "NAME_PORTFOLIO": pd.Categorical(["Cash", "POS", "Cash", "Cards"]),
            "NAME_PRODUCT_TYPE": pd.Categorical(["x-sell", "XNA", "walk-in", "XNA"]),
            "CNT_PAYMENT": [12.0, 0.0, 24.0, 0.0],
            "NAME_YIELD_GROUP": pd.Categorical(["high", "XNA", "low_normal", "middle"]),
            "NFLAG_INSURED_ON_APPROVAL": [1.0, np.nan, 0.0, 0.0],
        }
    )


def test_previous_features(previous_frame):
    out = previous.build_previous_features(previous_frame).set_index(ID)
    c = out.loc[100]
    assert c["previous__count"] == 3 and c["previous__approved_count"] == 1
    assert c["previous__approval_rate"] == pytest.approx(1 / 3)
    assert c["previous__refusal_rate"] == pytest.approx(1 / 3)
    assert c["previous__last_application_refused"] == 1.0  # most recent decision (-30) was refused
    assert c["previous__last_application_approved"] == 0.0
    assert c["previous__decisions_last_year"] == 2
    assert c["previous__amt_application_mean"] == pytest.approx(2000.0)  # zero application excluded
    assert c["previous__amt_credit_approved_mean"] == pytest.approx(1100.0)
    assert c["previous__cnt_payment_approved_mean"] == pytest.approx(12.0)
    assert c["previous__reject_scoring_count"] == 1 and c["previous__reject_hc_count"] == 0
    assert c["previous__cash_loan_share"] == pytest.approx(2 / 3)
    assert c["previous__client_new_share"] == pytest.approx(1 / 3)
    assert out.loc[200, "previous__last_application_approved"] == 1.0
    assert set(out.columns) == {f"previous__{s.name}" for s in previous.SPECS}


@pytest.fixture
def installment_rows():
    return pd.DataFrame(
        {
            "SK_ID_PREV": [1, 1, 1, 1, 2, 3],
            "SK_ID_CURR": [100, 100, 100, 100, 100, 200],
            "NUM_INSTALMENT_VERSION": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "NUM_INSTALMENT_NUMBER": [1, 2, 2, 3, 1, 1],
            "DAYS_INSTALMENT": [-100.0, -70.0, -70.0, -40.0, -400.0, -10.0],
            "DAYS_ENTRY_PAYMENT": [-95.0, -76.0, -60.0, np.nan, -410.0, -10.0],
            "AMT_INSTALMENT": [100.0, 200.0, 200.0, 100.0, 50.0, 0.0],
            "AMT_PAYMENT": [100.0, 120.0, 80.0, 0.0, 30.0, 10.0],
        }
    )


def test_installment_level_sums_split_payments_and_sign_convention(installment_rows):
    inst = installments.installment_level(installment_rows).set_index(
        ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"]
    )
    # installment 1: due -100, paid -95 -> 5 days LATE (positive)
    assert inst.loc[(1, 1), "days_late"] == 5.0 and inst.loc[(1, 1), "is_late"] == 1.0
    # installment 2: parts 120 + 80 = 200 = scheduled -> not under-paid;
    # the last part was paid at -60 -> 10 days late
    assert inst.loc[(1, 2), "paid"] == 200.0 and inst.loc[(1, 2), "is_underpaid"] == 0.0
    assert inst.loc[(1, 2), "n_parts"] == 2 and inst.loc[(1, 2), "is_split"] == 1.0
    assert inst.loc[(1, 2), "days_late"] == 10.0
    # installment 3: unpaid -> no lateness, under-paid, unpaid flag
    assert np.isnan(inst.loc[(1, 3), "days_late"]) and np.isnan(inst.loc[(1, 3), "is_late"])
    assert inst.loc[(1, 3), "is_unpaid"] == 1.0 and inst.loc[(1, 3), "is_underpaid"] == 1.0
    # early payment is negative (prev 2: due -400, paid -410); zero scheduled -> ratio undefined
    assert inst.loc[(2, 1), "days_late"] == -10.0 and inst.loc[(2, 1), "is_late"] == 0.0
    assert inst.loc[(3, 1), "scheduled"] == 0.0 and np.isnan(inst.loc[(3, 1), "payment_ratio"])


def test_installments_features(installment_rows):
    out = installments.build_installments_features(installment_rows).set_index(ID)
    c = out.loc[100]
    assert c["installments__n_installments"] == 4 and c["installments__n_contracts"] == 2
    assert c["installments__n_split_payments"] == 1
    assert c["installments__late_count"] == 2  # installments 1 and 2 paid late; 4 (prev 2) early
    assert c["installments__late_rate"] == pytest.approx(2 / 3)  # over the 3 paid installments
    assert c["installments__unpaid_rate"] == pytest.approx(1 / 4)
    assert c["installments__days_late_max"] == 10.0 and c["installments__days_late_min"] == -10.0
    assert c["installments__underpaid_rate"] == pytest.approx(
        2 / 4
    )  # installment 3 (0 of 100) and prev-2 (30 of 50)
    assert c["installments__paid_share"] == pytest.approx(330 / 450)
    assert c["installments__payment_gap_sum"] == pytest.approx(120.0)
    assert c["installments__n_installments_12m"] == 3
    assert c["installments__days_since_last_instalment"] == -40.0
    assert set(out.columns) == {f"installments__{s.name}" for s in installments.SPECS}


@pytest.fixture
def credit_card_frame():
    return pd.DataFrame(
        {
            "SK_ID_PREV": [1, 1, 1, 2],
            "SK_ID_CURR": [100, 100, 100, 200],
            "MONTHS_BALANCE": [-3, -2, -1, -1],
            "AMT_BALANCE": [500.0, 1200.0, -10.0, 0.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [1000.0, 1000.0, 1000.0, 0.0],
            "AMT_DRAWINGS_ATM_CURRENT": [100.0, 0.0, np.nan, 0.0],
            "AMT_DRAWINGS_CURRENT": [150.0, 0.0, 0.0, 0.0],
            "AMT_DRAWINGS_POS_CURRENT": [50.0, 0.0, np.nan, 0.0],
            "AMT_INST_MIN_REGULARITY": [50.0, 60.0, 0.0, 0.0],
            "AMT_PAYMENT_TOTAL_CURRENT": [50.0, 30.0, 0.0, 0.0],
            "CNT_DRAWINGS_ATM_CURRENT": [1.0, 0.0, np.nan, 0.0],
            "CNT_DRAWINGS_CURRENT": [2.0, 0.0, 0.0, 0.0],
            "CNT_INSTALMENT_MATURE_CUM": [1.0, 2.0, 3.0, 0.0],
            "NAME_CONTRACT_STATUS": pd.Categorical(["Active", "Active", "Completed", "Active"]),
            "SK_DPD": [0.0, 45.0, 0.0, 0.0],
            "SK_DPD_DEF": [0.0, 10.0, 0.0, 0.0],
        }
    )


def test_credit_card_features(credit_card_frame):
    out = credit_card.build_credit_card_features(credit_card_frame).set_index(ID)
    c = out.loc[100]
    assert c["credit_card__n_months"] == 3 and c["credit_card__n_cards"] == 1
    assert c["credit_card__utilization_mean"] == pytest.approx(
        (0.5 + 1.2 + 0.0) / 3
    )  # -10 clipped to 0
    assert c["credit_card__utilization_max"] == pytest.approx(1.2)
    assert c["credit_card__over_limit_rate"] == pytest.approx(1 / 3)
    assert c["credit_card__utilization_latest_mean"] == pytest.approx(0.0)  # month -1
    assert c["credit_card__balance_latest_mean"] == pytest.approx(-10.0)
    assert c["credit_card__dpd_max"] == 45.0 and c["credit_card__severe_dpd_rate"] == pytest.approx(
        1 / 3
    )
    assert c["credit_card__missed_min_payment_rate"] == pytest.approx(
        0.5
    )  # 30 < 60; 50 == 50 ok; min 0 ignored
    assert c["credit_card__atm_drawing_month_rate"] == pytest.approx(0.5)  # NaN month ignored
    assert c["credit_card__completed_cards"] == 1
    assert np.isnan(out.loc[200, "credit_card__utilization_mean"])  # zero limit -> undefined
    assert set(out.columns) == {f"credit_card__{s.name}" for s in credit_card.SPECS}


@pytest.fixture
def pos_frame():
    return pd.DataFrame(
        {
            "SK_ID_PREV": [1, 1, 2, 2, 3],
            "SK_ID_CURR": [100, 100, 100, 100, 200],
            "MONTHS_BALANCE": [-2, -1, -30, -29, -1],
            "CNT_INSTALMENT": [12.0, 12.0, 6.0, 6.0, 4.0],
            "CNT_INSTALMENT_FUTURE": [6.0, 5.0, 1.0, 0.0, 4.0],
            "NAME_CONTRACT_STATUS": pd.Categorical(
                ["Active", "Active", "Active", "Completed", "Signed"]
            ),
            "SK_DPD": [0.0, 40.0, 0.0, 0.0, 0.0],
            "SK_DPD_DEF": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_pos_features(pos_frame):
    out = pos_cash.build_pos_features(pos_frame).set_index(ID)
    c = out.loc[100]
    assert c["pos__n_months"] == 4 and c["pos__n_contracts"] == 2
    assert c["pos__dpd_positive_rate"] == pytest.approx(0.25)
    assert c["pos__severe_dpd_rate"] == pytest.approx(0.25)
    assert c["pos__recent_dpd_rate"] == pytest.approx(0.5)  # only months >= -12: (-2, -1)
    assert c["pos__completed_contracts"] == 1 and c["pos__completed_month_share"] == pytest.approx(
        0.25
    )
    assert c["pos__remaining_share_mean"] == pytest.approx((0.5 + 5 / 12 + 1 / 6 + 0.0) / 4)
    assert c["pos__months_max"] == -1 and c["pos__months_min"] == -30
    assert out.loc[200, "pos__active_month_share"] == 0.0
    assert set(out.columns) == {f"pos__{s.name}" for s in pos_cash.SPECS}


# --------------------------------------------------------------------------- #
# Grain and spec enforcement
# --------------------------------------------------------------------------- #
def test_finalize_rejects_spec_mismatch_and_duplicate_customers():
    specs = [FeatureSpec("a", "f", "d", "sum", True)]
    good = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([1, 2], name=ID))
    out = finalize_customer_table(good, prefix="x", specs=specs)
    assert list(out.columns) == [ID, "x__a"] and out["x__a"].dtype == np.float32
    with pytest.raises(ValueError, match="mismatch"):
        finalize_customer_table(good.assign(b=1.0), prefix="x", specs=specs)
    dup = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([1, 1], name=ID))
    with pytest.raises(ValueError, match="grain"):
        finalize_customer_table(dup, prefix="x", specs=specs)


def test_every_builder_returns_unique_customers(
    bureau_frames, previous_frame, installment_rows, credit_card_frame, pos_frame
):
    b, bb = bureau_frames
    tables = {
        "bureau": bureau.build_bureau_features(b, bb),
        "previous": previous.build_previous_features(previous_frame),
        "installments": installments.build_installments_features(installment_rows),
        "credit_card": credit_card.build_credit_card_features(credit_card_frame),
        "pos": pos_cash.build_pos_features(pos_frame),
    }
    for name, table in tables.items():
        assert table[ID].is_unique, name
        assert config.TARGET_COL not in table.columns
        assert all(
            c == ID or c.startswith(f"{assemble.SOURCE_MODULES[name].PREFIX}__")
            for c in table.columns
        )


# --------------------------------------------------------------------------- #
# Assembly (join safety, missing history) and catalog
# --------------------------------------------------------------------------- #
@pytest.fixture
def tables(bureau_frames, previous_frame, installment_rows, credit_card_frame, pos_frame):
    b, bb = bureau_frames
    return {
        "bureau": bureau.build_bureau_features(b, bb),
        "previous": previous.build_previous_features(previous_frame),
        "installments": installments.build_installments_features(installment_rows),
        "credit_card": credit_card.build_credit_card_features(credit_card_frame),
        "pos": pos_cash.build_pos_features(pos_frame),
    }


@pytest.fixture
def application():
    X = pd.DataFrame(
        {"AMT_INCOME_TOTAL": [1.0, 2.0, 3.0], "CODE_GENDER": ["F", "M", "F"]}, index=[7, 3, 9]
    )
    ids = pd.Series([300, 100, 200], index=X.index, name=ID)
    return X, ids


def test_build_feature_matrix_preserves_rows_and_marks_missing_history(application, tables):
    X, ids = application
    out = assemble.build_feature_matrix(X, ids, ["bureau", "installments"], tables=tables)
    assert len(out) == 3 and out.index.equals(X.index)
    assert (
        out.shape[1]
        == X.shape[1] + tables["bureau"].shape[1] - 1 + 1 + tables["installments"].shape[1] - 1 + 1
    )
    # customer 300 (row index 7) has no history anywhere
    assert out.loc[7, "bureau__has_history"] == 0.0 and out.loc[3, "bureau__has_history"] == 1.0
    assert out.loc[7, "bureau__credit_count"] == 0.0  # count-like: 0 is the true value
    assert np.isnan(out.loc[7, "bureau__credit_sum_mean"])  # mean: undefined without history
    assert (
        out.loc[7, "installments__has_history"] == 0.0
        and out.loc[7, "installments__n_installments"] == 0.0
    )
    assert out.loc[3, "bureau__credit_count"] == 2.0  # customer 100 aligned by id, not by position
    assert out.loc[9, "bureau__sold_or_bad_count"] == 1.0  # customer 200
    assert ID not in out.columns and config.TARGET_COL not in out.columns
    assert out["bureau__credit_count"].dtype == np.float32


def test_build_feature_matrix_rejects_bad_grain_and_identifiers(application, tables):
    X, ids = application
    duplicated = pd.concat([tables["pos"], tables["pos"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="not one row per"):
        assemble.build_feature_matrix(X, ids, ["pos"], tables={"pos": duplicated})
    with pytest.raises(ValueError, match="unique"):
        assemble.build_feature_matrix(
            X, pd.Series([100, 100, 200], index=X.index), ["pos"], tables=tables
        )
    with pytest.raises(ValueError, match="TARGET"):
        assemble.build_feature_matrix(X.assign(TARGET=0), ids, ["pos"], tables=tables)
    with pytest.raises(ValueError, match="same length"):
        assemble.build_feature_matrix(X, ids.iloc[:2], ["pos"], tables=tables)
    with pytest.raises(ValueError, match="Unknown"):
        assemble.build_feature_matrix(X, ids, ["nope"], tables=tables)
    with pytest.raises(ValueError, match="prefix"):
        assemble.build_feature_matrix(
            X,
            ids,
            ["pos"],
            tables={"pos": tables["pos"].rename(columns={"pos__n_months": "n_months"})},
        )


def test_feature_catalog_matches_generated_columns(tables):
    catalog = assemble.feature_catalog()
    assert catalog["feature"].is_unique
    for source, table in tables.items():
        produced = {c for c in table.columns if c != ID} | {assemble.has_column(source)}
        listed = set(catalog.loc[catalog["source"] == source, "feature"])
        assert produced == listed, source
    assert {
        "feature",
        "source",
        "family",
        "description",
        "aggregation",
        "leakage_assessment",
    } <= set(catalog.columns)
    assert catalog["description"].str.len().gt(5).all()
    assert catalog["leakage_assessment"].str.startswith("safe").all()


def test_feature_source_mapping():
    assert assemble.feature_source("bureau__credit_count") == "bureau"
    assert assemble.feature_source("credit_card__dpd_max") == "credit_card"
    assert assemble.feature_source("pos__has_history") == "pos"
    assert assemble.feature_source("AMT_INCOME_TOTAL") == "application"
    assert assemble.feature_source("EXT_SOURCE_1") == "application"
    srcs = assemble.sources_of(["AMT_CREDIT", "previous__count", "installments__late_rate"])
    assert list(srcs) == ["application", "previous", "installments"]


# --------------------------------------------------------------------------- #
# Cached build from tiny raw CSVs
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_raw_dir(
    tmp_path, bureau_frames, previous_frame, installment_rows, credit_card_frame, pos_frame
):
    raw = tmp_path / "raw"
    raw.mkdir()
    b, bb = bureau_frames
    prev = previous_frame.copy()
    for col in previous.EXCLUDED_TEMPORAL_COLUMNS:
        prev[col] = [365243.0, -10.0, np.nan, 5.0]
    frames = {
        "bureau": b,
        "bureau_balance": bb,
        "previous": prev,
        "installments": installment_rows,
        "credit_card": credit_card_frame,
        "pos": pos_frame,
    }
    for name, frame in frames.items():
        frame.to_csv(raw / config.RELATIONAL_TABLES[name]["filename"], index=False)
    return raw


def test_build_cli_builds_caches_and_writes_artifacts(tiny_raw_dir, tmp_path, capsys):
    processed, metrics = tmp_path / "processed", tmp_path / "metrics"
    args = [
        "--raw-dir",
        str(tiny_raw_dir),
        "--processed-dir",
        str(processed),
        "--metrics-dir",
        str(metrics),
    ]
    assert build_mod.main(args) == 0
    for source in assemble.SOURCES:
        table = pd.read_parquet(processed / f"{source}.parquet")
        assert (
            table[ID].is_unique and table.shape[1] == len(assemble.SOURCE_MODULES[source].SPECS) + 1
        )
    log = json.loads((metrics / "relational_build_log.json").read_text(encoding="utf-8"))
    assert set(log["sources"]) == set(assemble.SOURCES)
    assert all(log["last_run"]["status"][s] == "built" for s in assemble.SOURCES)
    entry = log["sources"]["installments"]
    assert entry["input_rows"] == {"installments_rows": 6} and entry["output_customers"] == 2
    assert {"seconds", "n_features", "output_memory_mb", "raw_files"} <= set(entry)
    audit = json.loads((metrics / "relational_data_audit.json").read_text(encoding="utf-8"))
    assert audit["installments"]["temporal"]["DAYS_INSTALMENT"]["n_positive"] == 0
    assert audit["previous"]["excluded_temporal_columns"] == list(
        previous.EXCLUDED_TEMPORAL_COLUMNS
    )
    catalog = pd.read_csv(metrics / "relational_feature_catalog.csv")
    assert len(catalog) == sum(len(m.SPECS) for m in assemble.SOURCE_MODULES.values()) + len(
        assemble.SOURCES
    )

    # Second run: everything is cached; --force rebuilds.
    assert build_mod.main(args) == 0
    log = json.loads((metrics / "relational_build_log.json").read_text(encoding="utf-8"))
    assert all(v == "cached" for v in log["last_run"]["status"].values())
    assert build_mod.main([*args, "--force", "--sources", "pos"]) == 0
    log = json.loads((metrics / "relational_build_log.json").read_text(encoding="utf-8"))
    assert log["last_run"]["status"] == {"pos": "built"}
    out = capsys.readouterr().out
    assert "Relational build" in out

    # The cached tables assemble onto application rows through the normal path.
    X = pd.DataFrame({"f": [1.0, 2.0]})
    ids = pd.Series([100, 300], name=ID)
    matrix = assemble.build_feature_matrix(X, ids, processed_dir=processed)
    assert (
        len(matrix) == 2
        and matrix.loc[1, "pos__has_history"] == 0.0
        and matrix.loc[0, "pos__n_months"] == 4.0
    )
