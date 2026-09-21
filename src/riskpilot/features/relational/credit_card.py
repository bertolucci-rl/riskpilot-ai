"""Customer-level features from ``credit_card_balance.csv`` (monthly card snapshots).

Grain (verified): one row per previous credit card ``SK_ID_PREV`` and month
(no duplicate (SK_ID_PREV, MONTHS_BALANCE) pairs); 3.84 M rows, 103,558
customers, 104,307 cards. ``MONTHS_BALANCE`` is in [-96, -1] relative to the
current application, so every snapshot predates it.

Conventions: utilization = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL is undefined
(NaN) on the 19.6 % of rows with a zero limit; negative balances (2,345 rows)
are clipped to 0 for the utilization only; receivable columns are not used
(``AMT_TOTAL_RECEIVABLE`` is negative on 109 k rows with no documented
meaning); drawings and payments are used as reported.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from riskpilot.features.relational.common import (
    FLOAT,
    ID,
    FeatureSpec,
    audit_series,
    finalize_customer_table,
    group_agg,
    latest_per_group,
    read_raw_table,
    safe_ratio,
    value_counts_dict,
)

SOURCE = "credit_card"
PREFIX = "credit_card"
LEAKAGE = (
    "safe: monthly snapshots with MONTHS_BALANCE in [-96, -1] relative to the current application"
)

USECOLS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_DRAWINGS_ATM_CURRENT",
    "AMT_DRAWINGS_CURRENT",
    "AMT_DRAWINGS_POS_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "AMT_PAYMENT_TOTAL_CURRENT",
    "CNT_DRAWINGS_ATM_CURRENT",
    "CNT_DRAWINGS_CURRENT",
    "CNT_INSTALMENT_MATURE_CUM",
    "NAME_CONTRACT_STATUS",
    "SK_DPD",
    "SK_DPD_DEF",
]
DTYPES = {
    "SK_ID_PREV": "int32",
    "SK_ID_CURR": "int32",
    "MONTHS_BALANCE": "int16",
    "AMT_BALANCE": "float32",
    "AMT_CREDIT_LIMIT_ACTUAL": "float32",
    "AMT_DRAWINGS_ATM_CURRENT": "float32",
    "AMT_DRAWINGS_CURRENT": "float32",
    "AMT_DRAWINGS_POS_CURRENT": "float32",
    "AMT_INST_MIN_REGULARITY": "float32",
    "AMT_PAYMENT_TOTAL_CURRENT": "float32",
    "CNT_DRAWINGS_ATM_CURRENT": "float32",
    "CNT_DRAWINGS_CURRENT": "float32",
    "CNT_INSTALMENT_MATURE_CUM": "float32",
    "NAME_CONTRACT_STATUS": "category",
    "SK_DPD": "float32",
    "SK_DPD_DEF": "float32",
}
HIGH_UTILIZATION = 0.8
SEVERE_DPD_DAYS = 30
PAYMENT_RATIO_CLIP = 5.0

SPECS: list[FeatureSpec] = [
    FeatureSpec("n_months", "volume", "monthly card records", "count", True),
    FeatureSpec("n_cards", "volume", "distinct previous credit cards", "nunique", True),
    FeatureSpec("months_min", "recency", "oldest month observed (MONTHS_BALANCE)", "min"),
    FeatureSpec("months_max", "recency", "most recent month observed", "max"),
    FeatureSpec("balance_mean", "balances", "mean monthly balance", "mean"),
    FeatureSpec("balance_max", "balances", "max monthly balance", "max"),
    FeatureSpec("balance_std", "balances", "std of monthly balance", "std"),
    FeatureSpec(
        "balance_latest_mean",
        "balances",
        "balance in the latest month of each card, averaged over cards",
        "last",
    ),
    FeatureSpec("limit_mean", "limits", "mean credit limit", "mean"),
    FeatureSpec("limit_max", "limits", "max credit limit", "max"),
    FeatureSpec("utilization_mean", "utilization", "mean balance / limit (limit > 0)", "mean"),
    FeatureSpec("utilization_max", "utilization", "max balance / limit", "max"),
    FeatureSpec(
        "high_utilization_rate", "utilization", "share of months with utilization > 0.8", "share"
    ),
    FeatureSpec("over_limit_rate", "utilization", "share of months with utilization > 1", "share"),
    FeatureSpec(
        "utilization_latest_mean",
        "utilization",
        "utilization in the latest month of each card, averaged",
        "last",
    ),
    FeatureSpec("dpd_mean", "delinquency", "mean SK_DPD", "mean"),
    FeatureSpec("dpd_max", "delinquency", "max SK_DPD", "max"),
    FeatureSpec("dpd_positive_rate", "delinquency", "share of months with SK_DPD > 0", "share"),
    FeatureSpec("severe_dpd_rate", "delinquency", "share of months with SK_DPD > 30", "share"),
    FeatureSpec("dpd_def_max", "delinquency", "max SK_DPD_DEF (with tolerance)", "max"),
    FeatureSpec("drawings_sum", "drawings", "total AMT_DRAWINGS_CURRENT", "sum", True),
    FeatureSpec("drawings_mean", "drawings", "mean monthly drawings", "mean"),
    FeatureSpec("atm_drawings_sum", "drawings", "total ATM drawings", "sum", True),
    FeatureSpec("pos_drawings_sum", "drawings", "total POS (goods) drawings", "sum", True),
    FeatureSpec(
        "atm_drawing_month_rate", "drawings", "share of months with an ATM drawing", "share"
    ),
    FeatureSpec("drawings_count_sum", "drawings", "total number of drawings", "sum", True),
    FeatureSpec("payment_mean", "payments", "mean AMT_PAYMENT_TOTAL_CURRENT", "mean"),
    FeatureSpec("min_payment_mean", "payments", "mean AMT_INST_MIN_REGULARITY", "mean"),
    FeatureSpec(
        "payment_to_min_ratio_mean",
        "payments",
        "mean payment / minimum due (min > 0, clipped to 5)",
        "mean",
    ),
    FeatureSpec(
        "missed_min_payment_rate",
        "payments",
        "share of months paying less than the minimum due",
        "share",
    ),
    FeatureSpec("active_month_share", "status", "share of months with status Active", "share"),
    FeatureSpec("completed_cards", "status", "cards with a Completed month", "count", True),
    FeatureSpec(
        "mature_installments_max",
        "status",
        "max CNT_INSTALMENT_MATURE_CUM (paid installments)",
        "max",
    ),
]


def build_credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    c = cc.copy(deep=False)
    balance_nonneg = c["AMT_BALANCE"].clip(lower=0)
    c["utilization"] = safe_ratio(balance_nonneg, c["AMT_CREDIT_LIMIT_ACTUAL"])
    c["high_util"] = (
        (c["utilization"] > HIGH_UTILIZATION).astype(FLOAT).where(c["utilization"].notna())
    )
    c["over_limit"] = (c["utilization"] > 1.0).astype(FLOAT).where(c["utilization"].notna())
    c["dpd_pos"] = (c["SK_DPD"] > 0).astype(FLOAT)
    c["dpd_severe"] = (c["SK_DPD"] > SEVERE_DPD_DAYS).astype(FLOAT)
    c["atm_month"] = (
        (c["CNT_DRAWINGS_ATM_CURRENT"] > 0)
        .astype(FLOAT)
        .where(c["CNT_DRAWINGS_ATM_CURRENT"].notna())
    )
    min_due = c["AMT_INST_MIN_REGULARITY"]
    c["payment_to_min"] = safe_ratio(c["AMT_PAYMENT_TOTAL_CURRENT"], min_due).clip(
        upper=PAYMENT_RATIO_CLIP
    )
    c["missed_min"] = (c["AMT_PAYMENT_TOTAL_CURRENT"] < min_due).astype(FLOAT).where(min_due > 0)
    status = c["NAME_CONTRACT_STATUS"].astype(str)
    c["active"] = (status == "Active").astype(FLOAT)
    c["completed"] = (status == "Completed").astype("int8")

    out = group_agg(
        c,
        ID,
        {
            "n_months": ("MONTHS_BALANCE", "size"),
            "n_cards": ("SK_ID_PREV", "nunique"),
            "months_min": ("MONTHS_BALANCE", "min"),
            "months_max": ("MONTHS_BALANCE", "max"),
            "balance_mean": ("AMT_BALANCE", "mean"),
            "balance_max": ("AMT_BALANCE", "max"),
            "balance_std": ("AMT_BALANCE", "std"),
            "limit_mean": ("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            "limit_max": ("AMT_CREDIT_LIMIT_ACTUAL", "max"),
            "utilization_mean": ("utilization", "mean"),
            "utilization_max": ("utilization", "max"),
            "high_utilization_rate": ("high_util", "mean"),
            "over_limit_rate": ("over_limit", "mean"),
            "dpd_mean": ("SK_DPD", "mean"),
            "dpd_max": ("SK_DPD", "max"),
            "dpd_positive_rate": ("dpd_pos", "mean"),
            "severe_dpd_rate": ("dpd_severe", "mean"),
            "dpd_def_max": ("SK_DPD_DEF", "max"),
            "drawings_sum": ("AMT_DRAWINGS_CURRENT", "sum"),
            "drawings_mean": ("AMT_DRAWINGS_CURRENT", "mean"),
            "atm_drawings_sum": ("AMT_DRAWINGS_ATM_CURRENT", "sum"),
            "pos_drawings_sum": ("AMT_DRAWINGS_POS_CURRENT", "sum"),
            "atm_drawing_month_rate": ("atm_month", "mean"),
            "drawings_count_sum": ("CNT_DRAWINGS_CURRENT", "sum"),
            "payment_mean": ("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
            "min_payment_mean": ("AMT_INST_MIN_REGULARITY", "mean"),
            "payment_to_min_ratio_mean": ("payment_to_min", "mean"),
            "missed_min_payment_rate": ("missed_min", "mean"),
            "active_month_share": ("active", "mean"),
            "mature_installments_max": ("CNT_INSTALMENT_MATURE_CUM", "max"),
        },
    )
    per_card_completed = c.groupby(["SK_ID_CURR", "SK_ID_PREV"], sort=False)["completed"].max()
    out["completed_cards"] = per_card_completed.groupby(level=0).sum().reindex(out.index)
    latest = latest_per_group(
        c[[ID, "SK_ID_PREV", "MONTHS_BALANCE", "AMT_BALANCE", "utilization"]],
        "SK_ID_PREV",
        "MONTHS_BALANCE",
    )
    latest_by_customer = latest.groupby(ID).agg(
        balance_latest_mean=("AMT_BALANCE", "mean"), utilization_latest_mean=("utilization", "mean")
    )
    out["balance_latest_mean"] = latest_by_customer["balance_latest_mean"].reindex(out.index)
    out["utilization_latest_mean"] = latest_by_customer["utilization_latest_mean"].reindex(
        out.index
    )
    return finalize_customer_table(out, prefix=PREFIX, specs=SPECS)


def audit_credit_card(cc: pd.DataFrame) -> dict[str, Any]:
    util = cc["AMT_BALANCE"].clip(lower=0) / cc["AMT_CREDIT_LIMIT_ACTUAL"].where(
        cc["AMT_CREDIT_LIMIT_ACTUAL"] > 0
    )
    return {
        "grain": {
            "n_rows": int(len(cc)),
            "n_customers": int(cc[ID].nunique()),
            "n_cards": int(cc["SK_ID_PREV"].nunique()),
            "duplicate_card_month_rows": int(cc.duplicated(["SK_ID_PREV", "MONTHS_BALANCE"]).sum()),
        },
        "temporal": {
            "months_min": int(cc["MONTHS_BALANCE"].min()),
            "months_max": int(cc["MONTHS_BALANCE"].max()),
        },
        "status": value_counts_dict(cc["NAME_CONTRACT_STATUS"]),
        "amounts": {
            c: audit_series(cc[c])
            for c in (
                "AMT_BALANCE",
                "AMT_CREDIT_LIMIT_ACTUAL",
                "AMT_DRAWINGS_CURRENT",
                "AMT_PAYMENT_TOTAL_CURRENT",
                "SK_DPD",
                "SK_DPD_DEF",
            )
        },
        "utilization": {
            "zero_limit_rows": int((cc["AMT_CREDIT_LIMIT_ACTUAL"] == 0).sum()),
            "median": float(util.median()),
            "q99": float(util.quantile(0.99)),
            "share_over_1": float((util > 1).mean()),
            "max": float(util.max()),
        },
        "anomalies": {
            "negative_balance_rows": int((cc["AMT_BALANCE"] < 0).sum()),
            "dpd_positive_share": float((cc["SK_DPD"] > 0).mean()),
            "dpd_def_positive_share": float((cc["SK_DPD_DEF"] > 0).mean()),
        },
    }


def load_and_build(
    *, raw_dir=None, nrows: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    kwargs = {"nrows": nrows}
    if raw_dir is not None:
        kwargs["raw_dir"] = raw_dir
    cc = read_raw_table("credit_card", usecols=USECOLS, dtypes=DTYPES, **kwargs)
    audit = audit_credit_card(cc)
    features = build_credit_card_features(cc)
    sizes = {"credit_card_rows": int(len(cc))}
    del cc
    return features, audit, sizes
