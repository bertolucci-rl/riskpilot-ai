"""Customer-level features from ``POS_CASH_balance.csv`` (monthly POS / cash-loan snapshots).

Grain (verified): one row per previous POS/cash credit ``SK_ID_PREV`` and
month (no duplicate pairs); 10.0 M rows, 337,252 customers, 936,325 credits.
``MONTHS_BALANCE`` is in [-96, -1] relative to the current application.

``CNT_INSTALMENT`` is the term (can change), ``CNT_INSTALMENT_FUTURE`` the
installments left; 7,420 rows have more left than the term (re-scheduling),
used as reported. ``SK_DPD`` / ``SK_DPD_DEF`` are days past due in the month
(``_DEF`` ignores small debts).
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
    read_raw_table,
    safe_ratio,
    value_counts_dict,
)
from riskpilot.features.relational.temporal import require_historical

SOURCE = "pos"
PREFIX = "pos"
LEAKAGE = (
    "safe: monthly snapshots with MONTHS_BALANCE in [-96, -1] relative to the current application"
)

USECOLS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "CNT_INSTALMENT",
    "CNT_INSTALMENT_FUTURE",
    "NAME_CONTRACT_STATUS",
    "SK_DPD",
    "SK_DPD_DEF",
]
DTYPES = {
    "SK_ID_PREV": "int32",
    "SK_ID_CURR": "int32",
    "MONTHS_BALANCE": "int16",
    "CNT_INSTALMENT": "float32",
    "CNT_INSTALMENT_FUTURE": "float32",
    "NAME_CONTRACT_STATUS": "category",
    "SK_DPD": "float32",
    "SK_DPD_DEF": "float32",
}
SEVERE_DPD_DAYS = 30
RECENT_MONTHS = 12

SPECS: list[FeatureSpec] = [
    FeatureSpec("n_months", "volume", "monthly POS/cash records", "count", True),
    FeatureSpec("n_contracts", "volume", "distinct previous POS/cash credits", "nunique", True),
    FeatureSpec("months_min", "recency", "oldest month observed", "min"),
    FeatureSpec("months_max", "recency", "most recent month observed", "max"),
    FeatureSpec("cnt_instalment_mean", "terms", "mean term (CNT_INSTALMENT)", "mean"),
    FeatureSpec("cnt_instalment_max", "terms", "max term", "max"),
    FeatureSpec("instalments_future_mean", "terms", "mean installments left", "mean"),
    FeatureSpec("instalments_future_max", "terms", "max installments left", "max"),
    FeatureSpec("remaining_share_mean", "terms", "mean installments left / term", "mean"),
    FeatureSpec("dpd_mean", "delinquency", "mean SK_DPD", "mean"),
    FeatureSpec("dpd_max", "delinquency", "max SK_DPD", "max"),
    FeatureSpec("dpd_positive_rate", "delinquency", "share of months with SK_DPD > 0", "share"),
    FeatureSpec("severe_dpd_rate", "delinquency", "share of months with SK_DPD > 30", "share"),
    FeatureSpec("dpd_def_max", "delinquency", "max SK_DPD_DEF", "max"),
    FeatureSpec(
        "dpd_def_positive_rate", "delinquency", "share of months with SK_DPD_DEF > 0", "share"
    ),
    FeatureSpec(
        "recent_dpd_rate", "delinquency", "share of months in the last 12 with SK_DPD > 0", "share"
    ),
    FeatureSpec("active_month_share", "status", "share of months with status Active", "share"),
    FeatureSpec(
        "completed_month_share", "status", "share of months with status Completed", "share"
    ),
    FeatureSpec("completed_contracts", "status", "credits with a Completed month", "count", True),
]


def build_pos_features(pos: pd.DataFrame) -> pd.DataFrame:
    require_historical(pos, "MONTHS_BALANCE")
    p = pos.copy(deep=False)
    p["dpd_pos"] = (p["SK_DPD"] > 0).astype(FLOAT)
    p["dpd_severe"] = (p["SK_DPD"] > SEVERE_DPD_DAYS).astype(FLOAT)
    p["dpd_def_pos"] = (p["SK_DPD_DEF"] > 0).astype(FLOAT)
    p["recent_dpd"] = p["dpd_pos"].where(p["MONTHS_BALANCE"] >= -RECENT_MONTHS)
    p["remaining_share"] = safe_ratio(p["CNT_INSTALMENT_FUTURE"], p["CNT_INSTALMENT"])
    status = p["NAME_CONTRACT_STATUS"].astype(str)
    p["active"] = (status == "Active").astype(FLOAT)
    p["completed"] = (status == "Completed").astype("int8")
    out = group_agg(
        p,
        ID,
        {
            "n_months": ("MONTHS_BALANCE", "size"),
            "n_contracts": ("SK_ID_PREV", "nunique"),
            "months_min": ("MONTHS_BALANCE", "min"),
            "months_max": ("MONTHS_BALANCE", "max"),
            "cnt_instalment_mean": ("CNT_INSTALMENT", "mean"),
            "cnt_instalment_max": ("CNT_INSTALMENT", "max"),
            "instalments_future_mean": ("CNT_INSTALMENT_FUTURE", "mean"),
            "instalments_future_max": ("CNT_INSTALMENT_FUTURE", "max"),
            "remaining_share_mean": ("remaining_share", "mean"),
            "dpd_mean": ("SK_DPD", "mean"),
            "dpd_max": ("SK_DPD", "max"),
            "dpd_positive_rate": ("dpd_pos", "mean"),
            "severe_dpd_rate": ("dpd_severe", "mean"),
            "dpd_def_max": ("SK_DPD_DEF", "max"),
            "dpd_def_positive_rate": ("dpd_def_pos", "mean"),
            "recent_dpd_rate": ("recent_dpd", "mean"),
            "active_month_share": ("active", "mean"),
            "completed_month_share": ("completed", "mean"),
        },
    )
    per_contract_completed = p.groupby(["SK_ID_CURR", "SK_ID_PREV"], sort=False)["completed"].max()
    out["completed_contracts"] = per_contract_completed.groupby(level=0).sum().reindex(out.index)
    return finalize_customer_table(out, prefix=PREFIX, specs=SPECS)


def audit_pos(pos: pd.DataFrame) -> dict[str, Any]:
    return {
        "grain": {
            "n_rows": int(len(pos)),
            "n_customers": int(pos[ID].nunique()),
            "n_contracts": int(pos["SK_ID_PREV"].nunique()),
            "duplicate_contract_month_rows": int(
                pos.duplicated(["SK_ID_PREV", "MONTHS_BALANCE"]).sum()
            ),
        },
        "temporal": {
            "months_min": int(pos["MONTHS_BALANCE"].min()),
            "months_max": int(pos["MONTHS_BALANCE"].max()),
        },
        "status": value_counts_dict(pos["NAME_CONTRACT_STATUS"]),
        "counts": {
            c: audit_series(pos[c])
            for c in ("CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF")
        },
        "anomalies": {
            "future_exceeds_term_rows": int(
                (pos["CNT_INSTALMENT_FUTURE"] > pos["CNT_INSTALMENT"]).sum()
            ),
            "dpd_positive_share": float((pos["SK_DPD"] > 0).mean()),
            "dpd_def_positive_share": float((pos["SK_DPD_DEF"] > 0).mean()),
        },
    }


def load_and_build(
    *, raw_dir=None, nrows: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    kwargs = {"nrows": nrows}
    if raw_dir is not None:
        kwargs["raw_dir"] = raw_dir
    pos = read_raw_table("pos", usecols=USECOLS, dtypes=DTYPES, **kwargs)
    audit = audit_pos(pos)
    features = build_pos_features(pos)
    sizes = {"pos_rows": int(len(pos))}
    del pos
    return features, audit, sizes
