"""Customer-level features from ``bureau.csv`` and ``bureau_balance.csv``.

Grain (verified in the relational audit): ``bureau`` has one row per external
credit ``SK_ID_BUREAU`` (unique), many per ``SK_ID_CURR``; ``bureau_balance``
has one row per (``SK_ID_BUREAU``, ``MONTHS_BALANCE``) with ``MONTHS_BALANCE``
in [-96, 0] relative to the current application. Only 774,354 of the
1,716,428 bureau credits (45 %) have monthly history; 43,041 monthly-history
credits have no row in ``bureau`` and are dropped by the validated
many-to-one merge (a documented discrepancy of the dataset).

Temporal semantics (official column dictionary): every ``DAYS_*`` column is
"relative to the current application"; ``DAYS_CREDIT``, ``DAYS_ENDDATE_FACT``
and ``DAYS_CREDIT_UPDATE`` describe the past (``<= 0``; 17 rows of
``DAYS_CREDIT_UPDATE`` are slightly positive and are clipped to 0),
``DAYS_CREDIT_ENDDATE`` is the *remaining* planned duration at application
time (positive for credits that end later, which is known at application),
and the amounts and ``CREDIT_DAY_OVERDUE`` are "at the time of application".
Nothing in either table postdates the application, so every feature here is
historical information available at decision time.

Pipeline: aggregate ``bureau_balance`` to one row per ``SK_ID_BUREAU``, merge
that summary into ``bureau`` (many-to-one, validated), then aggregate to one
row per ``SK_ID_CURR``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from riskpilot.features.relational.common import (
    ID,
    FeatureSpec,
    audit_series,
    finalize_customer_table,
    group_agg,
    read_raw_table,
    safe_ratio,
    value_counts_dict,
)

SOURCE = "bureau"
PREFIX = "bureau"
LEAKAGE = (
    "safe: DAYS_* are relative to the current application (<= 0 except the planned "
    "DAYS_CREDIT_ENDDATE), amounts/overdue are 'at the time of application', "
    "bureau_balance months are in [-96, 0]"
)

BUREAU_USECOLS = [
    "SK_ID_CURR",
    "SK_ID_BUREAU",
    "CREDIT_ACTIVE",
    "CREDIT_TYPE",
    "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "AMT_CREDIT_MAX_OVERDUE",
    "CNT_CREDIT_PROLONG",
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_LIMIT",
    "AMT_CREDIT_SUM_OVERDUE",
    "DAYS_CREDIT_UPDATE",
    "AMT_ANNUITY",
]
BUREAU_DTYPES = {
    "SK_ID_CURR": "int32",
    "SK_ID_BUREAU": "int32",
    "CREDIT_ACTIVE": "category",
    "CREDIT_TYPE": "category",
    "DAYS_CREDIT": "float32",
    "CREDIT_DAY_OVERDUE": "float32",
    "DAYS_CREDIT_ENDDATE": "float32",
    "DAYS_ENDDATE_FACT": "float32",
    "AMT_CREDIT_MAX_OVERDUE": "float32",
    "CNT_CREDIT_PROLONG": "float32",
    "AMT_CREDIT_SUM": "float32",
    "AMT_CREDIT_SUM_DEBT": "float32",
    "AMT_CREDIT_SUM_LIMIT": "float32",
    "AMT_CREDIT_SUM_OVERDUE": "float32",
    "DAYS_CREDIT_UPDATE": "float32",
    "AMT_ANNUITY": "float32",
}
BALANCE_USECOLS = ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]
BALANCE_DTYPES = {"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"}

# STATUS: C closed, X unknown, 0 no DPD, 1..5 increasing delinquency
# (5 = 120+ days past due, sold or written off)
DPD_SEVERITY = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
CREDIT_TYPE_GROUPS = {
    "Consumer credit": "consumer",
    "Credit card": "credit_card",
    "Car loan": "car_loan",
    "Mortgage": "mortgage",
    "Microloan": "microloan",
}
RECENT_DAYS = 365
RECENT_MONTHS = 12

SPECS: list[FeatureSpec] = [
    FeatureSpec("credit_count", "counts", "number of external credits reported", "count", True),
    FeatureSpec("active_count", "counts", "credits with CREDIT_ACTIVE == Active", "count", True),
    FeatureSpec("closed_count", "counts", "credits with CREDIT_ACTIVE == Closed", "count", True),
    FeatureSpec("sold_or_bad_count", "counts", "credits Sold or Bad debt", "count", True),
    FeatureSpec("active_ratio", "counts", "active credits / all credits", "ratio"),
    FeatureSpec("n_credit_types", "counts", "distinct CREDIT_TYPE values", "nunique", True),
    FeatureSpec("consumer_count", "credit types", "Consumer credit records", "count", True),
    FeatureSpec("credit_card_count", "credit types", "Credit card records", "count", True),
    FeatureSpec("car_loan_count", "credit types", "Car loan records", "count", True),
    FeatureSpec("mortgage_count", "credit types", "Mortgage records", "count", True),
    FeatureSpec("microloan_count", "credit types", "Microloan records", "count", True),
    FeatureSpec(
        "other_type_count", "credit types", "records of any other credit type", "count", True
    ),
    FeatureSpec(
        "credits_last_year", "recency", "credits opened in the last 365 days", "count", True
    ),
    FeatureSpec(
        "credit_sum_total", "amounts", "sum of AMT_CREDIT_SUM (current credit amounts)", "sum", True
    ),
    FeatureSpec("credit_sum_mean", "amounts", "mean AMT_CREDIT_SUM", "mean"),
    FeatureSpec("credit_sum_max", "amounts", "max AMT_CREDIT_SUM", "max"),
    FeatureSpec(
        "debt_sum_total",
        "amounts",
        "sum of AMT_CREDIT_SUM_DEBT (negatives clipped to 0)",
        "sum",
        True,
    ),
    FeatureSpec("debt_sum_mean", "amounts", "mean AMT_CREDIT_SUM_DEBT", "mean"),
    FeatureSpec("debt_sum_max", "amounts", "max AMT_CREDIT_SUM_DEBT", "max"),
    FeatureSpec("active_debt_sum", "amounts", "sum of debt on active credits", "sum", True),
    FeatureSpec(
        "limit_sum",
        "amounts",
        "sum of AMT_CREDIT_SUM_LIMIT (card limits, negatives clipped)",
        "sum",
        True,
    ),
    FeatureSpec("annuity_sum", "amounts", "sum of reported annuities", "sum", True),
    FeatureSpec("overdue_sum_total", "delinquency", "sum of AMT_CREDIT_SUM_OVERDUE", "sum", True),
    FeatureSpec("overdue_sum_max", "delinquency", "max AMT_CREDIT_SUM_OVERDUE", "max"),
    FeatureSpec(
        "max_overdue_max",
        "delinquency",
        "max of AMT_CREDIT_MAX_OVERDUE (worst ever overdue)",
        "max",
    ),
    FeatureSpec("max_overdue_mean", "delinquency", "mean AMT_CREDIT_MAX_OVERDUE", "mean"),
    FeatureSpec("day_overdue_max", "delinquency", "max CREDIT_DAY_OVERDUE at application", "max"),
    FeatureSpec("day_overdue_mean", "delinquency", "mean CREDIT_DAY_OVERDUE", "mean"),
    FeatureSpec(
        "overdue_credit_count", "delinquency", "credits with CREDIT_DAY_OVERDUE > 0", "count", True
    ),
    FeatureSpec("prolong_sum", "delinquency", "sum of CNT_CREDIT_PROLONG", "sum", True),
    FeatureSpec("debt_to_credit_ratio", "ratios", "total debt / total credit amount", "ratio"),
    FeatureSpec("overdue_to_debt_ratio", "ratios", "total overdue / total debt", "ratio"),
    FeatureSpec(
        "days_credit_max", "recency", "DAYS_CREDIT of the most recent credit (closest to 0)", "max"
    ),
    FeatureSpec("days_credit_min", "recency", "DAYS_CREDIT of the oldest credit", "min"),
    FeatureSpec("days_credit_mean", "recency", "mean DAYS_CREDIT", "mean"),
    FeatureSpec(
        "days_update_max", "recency", "most recent DAYS_CREDIT_UPDATE (clipped <= 0)", "max"
    ),
    FeatureSpec(
        "enddate_remaining_max",
        "recency",
        "max planned remaining duration of active credits",
        "max",
    ),
    FeatureSpec(
        "days_enddate_fact_max", "recency", "most recently closed credit (DAYS_ENDDATE_FACT)", "max"
    ),
    FeatureSpec(
        "bb_credits_with_history",
        "monthly history",
        "credits that appear in bureau_balance",
        "count",
        True,
    ),
    FeatureSpec(
        "bb_months_sum", "monthly history", "monthly records over all credits", "sum", True
    ),
    FeatureSpec(
        "bb_history_max", "monthly history", "longest monthly history of one credit (months)", "max"
    ),
    FeatureSpec(
        "bb_dpd_months_sum",
        "monthly history",
        "months in DPD status (1-5) over all credits",
        "sum",
        True,
    ),
    FeatureSpec(
        "bb_dpd_share_mean",
        "monthly history",
        "mean over credits of DPD months / known-status months",
        "mean",
    ),
    FeatureSpec("bb_status_max", "monthly history", "worst status severity ever (0-5)", "max"),
    FeatureSpec(
        "bb_severe_credits", "monthly history", "credits ever in status 3+ (61+ DPD)", "count", True
    ),
    FeatureSpec(
        "bb_latest_dpd_credits",
        "monthly history",
        "credits whose latest known status is DPD",
        "count",
        True,
    ),
    FeatureSpec(
        "bb_recent_dpd_credits",
        "monthly history",
        "credits with a DPD month in the last 12 months",
        "count",
        True,
    ),
    FeatureSpec(
        "bb_closed_share_mean",
        "monthly history",
        "mean over credits of closed months share",
        "mean",
    ),
]


def aggregate_bureau_balance(balance: pd.DataFrame) -> pd.DataFrame:
    """One row per ``SK_ID_BUREAU`` from the monthly status records."""
    b = balance
    severity = b["STATUS"].astype(str).map(DPD_SEVERITY).fillna(0).astype("int8")
    known = ~b["STATUS"].isin(["X"])
    frame = pd.DataFrame(
        {
            "SK_ID_BUREAU": b["SK_ID_BUREAU"].to_numpy(),
            "months": np.int8(1),
            "known": known.to_numpy(),
            "dpd": (severity > 0).to_numpy(),
            "severity": severity.to_numpy(),
            "closed": (b["STATUS"] == "C").to_numpy(),
            "recent_dpd": ((severity > 0) & (b["MONTHS_BALANCE"] >= -RECENT_MONTHS)).to_numpy(),
            "month": b["MONTHS_BALANCE"].to_numpy(),
        }
    )
    g = frame.groupby("SK_ID_BUREAU", sort=False)
    out = g.agg(
        bb_months=("months", "sum"),
        bb_known_months=("known", "sum"),
        bb_dpd_months=("dpd", "sum"),
        bb_status_max=("severity", "max"),
        bb_closed_months=("closed", "sum"),
        bb_recent_dpd=("recent_dpd", "max"),
        bb_oldest_month=("month", "min"),
    )
    latest_idx = g["month"].idxmax()
    latest = frame.loc[latest_idx, ["SK_ID_BUREAU", "severity"]].set_index("SK_ID_BUREAU")
    out["bb_latest_dpd"] = (latest["severity"] > 0).reindex(out.index).fillna(False)
    out["bb_dpd_share"] = safe_ratio(out["bb_dpd_months"], out["bb_known_months"])
    out["bb_closed_share"] = safe_ratio(out["bb_closed_months"], out["bb_months"])
    out["bb_history_len"] = (-out["bb_oldest_month"]).astype("int16") + 1
    out["bb_severe"] = out["bb_status_max"] >= 3
    return out.reset_index()


def build_bureau_features(bureau: pd.DataFrame, balance: pd.DataFrame | None) -> pd.DataFrame:
    """One row per ``SK_ID_CURR`` from ``bureau`` (+ optional ``bureau_balance``)."""
    b = bureau.copy(deep=False)
    b["debt"] = b["AMT_CREDIT_SUM_DEBT"].clip(lower=0)
    b["limit"] = b["AMT_CREDIT_SUM_LIMIT"].clip(lower=0)
    b["is_active"] = (b["CREDIT_ACTIVE"] == "Active").astype("int8")
    b["is_closed"] = (b["CREDIT_ACTIVE"] == "Closed").astype("int8")
    b["is_sold_or_bad"] = b["CREDIT_ACTIVE"].isin(["Sold", "Bad debt"]).astype("int8")
    b["active_debt"] = b["debt"].where(b["is_active"] == 1, 0.0)
    type_group = b["CREDIT_TYPE"].astype(str).map(CREDIT_TYPE_GROUPS).fillna("other")
    for group in [*CREDIT_TYPE_GROUPS.values(), "other"]:
        b[f"type_{group}"] = (type_group == group).astype("int8")
    b["recent_credit"] = (b["DAYS_CREDIT"] >= -RECENT_DAYS).astype("int8")
    b["overdue_credit"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype("int8")
    b["days_update"] = b["DAYS_CREDIT_UPDATE"].clip(upper=0)
    b["active_enddate"] = b["DAYS_CREDIT_ENDDATE"].where(b["is_active"] == 1)

    if balance is not None and len(balance):
        summary = aggregate_bureau_balance(balance)
        b = b.merge(summary, on="SK_ID_BUREAU", how="left", validate="many_to_one")
    else:
        for col in [
            "bb_months",
            "bb_dpd_months",
            "bb_status_max",
            "bb_recent_dpd",
            "bb_latest_dpd",
            "bb_dpd_share",
            "bb_closed_share",
            "bb_history_len",
            "bb_severe",
        ]:
            b[col] = np.nan
    b["bb_has_history"] = b["bb_months"].notna().astype("int8")
    for col in ("bb_recent_dpd", "bb_latest_dpd", "bb_severe"):
        b[col] = b[col].astype("float32")

    out = group_agg(
        b,
        ID,
        {
            "credit_count": ("SK_ID_BUREAU", "size"),
            "active_count": ("is_active", "sum"),
            "closed_count": ("is_closed", "sum"),
            "sold_or_bad_count": ("is_sold_or_bad", "sum"),
            "n_credit_types": ("CREDIT_TYPE", "nunique"),
            "consumer_count": ("type_consumer", "sum"),
            "credit_card_count": ("type_credit_card", "sum"),
            "car_loan_count": ("type_car_loan", "sum"),
            "mortgage_count": ("type_mortgage", "sum"),
            "microloan_count": ("type_microloan", "sum"),
            "other_type_count": ("type_other", "sum"),
            "credits_last_year": ("recent_credit", "sum"),
            "credit_sum_total": ("AMT_CREDIT_SUM", "sum"),
            "credit_sum_mean": ("AMT_CREDIT_SUM", "mean"),
            "credit_sum_max": ("AMT_CREDIT_SUM", "max"),
            "debt_sum_total": ("debt", "sum"),
            "debt_sum_mean": ("debt", "mean"),
            "debt_sum_max": ("debt", "max"),
            "active_debt_sum": ("active_debt", "sum"),
            "limit_sum": ("limit", "sum"),
            "annuity_sum": ("AMT_ANNUITY", "sum"),
            "overdue_sum_total": ("AMT_CREDIT_SUM_OVERDUE", "sum"),
            "overdue_sum_max": ("AMT_CREDIT_SUM_OVERDUE", "max"),
            "max_overdue_max": ("AMT_CREDIT_MAX_OVERDUE", "max"),
            "max_overdue_mean": ("AMT_CREDIT_MAX_OVERDUE", "mean"),
            "day_overdue_max": ("CREDIT_DAY_OVERDUE", "max"),
            "day_overdue_mean": ("CREDIT_DAY_OVERDUE", "mean"),
            "overdue_credit_count": ("overdue_credit", "sum"),
            "prolong_sum": ("CNT_CREDIT_PROLONG", "sum"),
            "days_credit_max": ("DAYS_CREDIT", "max"),
            "days_credit_min": ("DAYS_CREDIT", "min"),
            "days_credit_mean": ("DAYS_CREDIT", "mean"),
            "days_update_max": ("days_update", "max"),
            "enddate_remaining_max": ("active_enddate", "max"),
            "days_enddate_fact_max": ("DAYS_ENDDATE_FACT", "max"),
            "bb_credits_with_history": ("bb_has_history", "sum"),
            "bb_months_sum": ("bb_months", "sum"),
            "bb_history_max": ("bb_history_len", "max"),
            "bb_dpd_months_sum": ("bb_dpd_months", "sum"),
            "bb_dpd_share_mean": ("bb_dpd_share", "mean"),
            "bb_status_max": ("bb_status_max", "max"),
            "bb_severe_credits": ("bb_severe", "sum"),
            "bb_latest_dpd_credits": ("bb_latest_dpd", "sum"),
            "bb_recent_dpd_credits": ("bb_recent_dpd", "sum"),
            "bb_closed_share_mean": ("bb_closed_share", "mean"),
        },
    )
    out["active_ratio"] = safe_ratio(out["active_count"], out["credit_count"])
    out["debt_to_credit_ratio"] = safe_ratio(out["debt_sum_total"], out["credit_sum_total"])
    out["overdue_to_debt_ratio"] = safe_ratio(out["overdue_sum_total"], out["debt_sum_total"])
    return finalize_customer_table(out, prefix=PREFIX, specs=SPECS)


def audit_bureau(bureau: pd.DataFrame, balance: pd.DataFrame | None) -> dict[str, Any]:
    """Data-quality observations (reported, never silently fixed)."""
    audit: dict[str, Any] = {
        "grain": {
            "n_rows": int(len(bureau)),
            "n_customers": int(bureau[ID].nunique()),
            "sk_id_bureau_unique": bool(bureau["SK_ID_BUREAU"].is_unique),
        },
        "temporal": {
            c: {
                "min": float(bureau[c].min()),
                "max": float(bureau[c].max()),
                "n_positive": int((bureau[c] > 0).sum()),
            }
            for c in (
                "DAYS_CREDIT",
                "DAYS_CREDIT_ENDDATE",
                "DAYS_ENDDATE_FACT",
                "DAYS_CREDIT_UPDATE",
            )
        },
        "credit_active": value_counts_dict(bureau["CREDIT_ACTIVE"]),
        "credit_type": value_counts_dict(bureau["CREDIT_TYPE"], top=15),
        "amounts": {
            c: audit_series(bureau[c])
            for c in (
                "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_SUM_LIMIT",
                "AMT_CREDIT_SUM_OVERDUE",
                "AMT_CREDIT_MAX_OVERDUE",
                "CREDIT_DAY_OVERDUE",
            )
        },
        "anomalies": {
            "debt_exceeds_credit_sum_rows": int(
                (bureau["AMT_CREDIT_SUM_DEBT"] > bureau["AMT_CREDIT_SUM"]).sum()
            ),
            "negative_debt_rows_clipped": int((bureau["AMT_CREDIT_SUM_DEBT"] < 0).sum()),
            "negative_limit_rows_clipped": int((bureau["AMT_CREDIT_SUM_LIMIT"] < 0).sum()),
            "positive_days_credit_update_rows_clipped": int(
                (bureau["DAYS_CREDIT_UPDATE"] > 0).sum()
            ),
            "active_with_past_enddate_rows": int(
                ((bureau["CREDIT_ACTIVE"] == "Active") & (bureau["DAYS_CREDIT_ENDDATE"] < 0)).sum()
            ),
        },
    }
    if balance is not None:
        audit["bureau_balance"] = {
            "n_rows": int(len(balance)),
            "n_credits": int(balance["SK_ID_BUREAU"].nunique()),
            "credits_in_bureau_with_history": int(
                bureau["SK_ID_BUREAU"].isin(balance["SK_ID_BUREAU"].unique()).sum()
            ),
            "months_min": int(balance["MONTHS_BALANCE"].min()),
            "months_max": int(balance["MONTHS_BALANCE"].max()),
            "duplicate_credit_month_rows": int(
                balance.duplicated(["SK_ID_BUREAU", "MONTHS_BALANCE"]).sum()
            ),
            "status_counts": value_counts_dict(balance["STATUS"]),
        }
    return audit


def load_and_build(
    *, raw_dir=None, nrows: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    """Read the raw tables, build the customer table and the audit; return input sizes."""
    kwargs = {"nrows": nrows}
    if raw_dir is not None:
        kwargs["raw_dir"] = raw_dir
    bureau = read_raw_table("bureau", usecols=BUREAU_USECOLS, dtypes=BUREAU_DTYPES, **kwargs)
    balance = read_raw_table(
        "bureau_balance", usecols=BALANCE_USECOLS, dtypes=BALANCE_DTYPES, **kwargs
    )
    audit = audit_bureau(bureau, balance)
    features = build_bureau_features(bureau, balance)
    sizes = {"bureau_rows": int(len(bureau)), "bureau_balance_rows": int(len(balance))}
    del bureau, balance
    return features, audit, sizes
