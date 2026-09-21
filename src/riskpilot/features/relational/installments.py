"""Customer-level payment-behaviour features from ``installments_payments.csv``.

Grain (verified): 13.6 M rows, one per (previous credit ``SK_ID_PREV``,
``NUM_INSTALMENT_NUMBER``, ``NUM_INSTALMENT_VERSION``) **and payment part**:
4.8 % of the rows share a key with another row because an installment was paid
in several parts. Payment parts are therefore summed to one row per
installment before anything is aggregated per customer, so a split payment is
not mistaken for an under-payment.

Temporal semantics: ``DAYS_INSTALMENT`` (when the installment was due) and
``DAYS_ENTRY_PAYMENT`` (when it was actually paid) are both relative to the
current application date and both are in [-4921, -1] on every row: every
installment in the table was due and (if paid) paid before the current
application. ``days_late = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`` is positive
when the payment came after the due date (verified: due -100, paid -95 ->
+5 days late); the median installment is paid 6 days early.

Conventions: ``payment_ratio = paid / scheduled`` is undefined (NaN) when the
scheduled amount is 0 and clipped to [0, 2] for the mean; an installment is
under-paid when ``paid < scheduled - 1`` (one currency unit of tolerance),
over-paid when ``paid > scheduled + 1``; an installment with no
``DAYS_ENTRY_PAYMENT`` (2,905 rows) is unpaid and has no lateness value.
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
)
from riskpilot.features.relational.temporal import require_historical

SOURCE = "installments"
PREFIX = "installments"
LEAKAGE = (
    "safe: DAYS_INSTALMENT and DAYS_ENTRY_PAYMENT are relative to the current application "
    "and both <= -1 on every row; lateness = entry - instalment (positive = late)"
)

USECOLS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
]
DTYPES = {
    "SK_ID_PREV": "int32",
    "SK_ID_CURR": "int32",
    "NUM_INSTALMENT_VERSION": "float32",
    "NUM_INSTALMENT_NUMBER": "int16",
    "DAYS_INSTALMENT": "float32",
    "DAYS_ENTRY_PAYMENT": "float32",
    "AMT_INSTALMENT": "float32",
    "AMT_PAYMENT": "float32",
}
INSTALLMENT_KEY = ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"]
AMOUNT_TOLERANCE = 1.0
SEVERE_LATE_DAYS = 30
RECENT_DAYS = 365
RATIO_CLIP = 2.0

SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "n_installments", "volume", "installments (after summing split payments)", "count", True
    ),
    FeatureSpec(
        "n_contracts", "volume", "distinct previous credits with installments", "nunique", True
    ),
    FeatureSpec(
        "n_split_payments", "volume", "installments paid in more than one part", "count", True
    ),
    FeatureSpec(
        "days_since_last_instalment",
        "recency",
        "DAYS_INSTALMENT of the most recent installment",
        "max",
    ),
    FeatureSpec("history_span_days", "recency", "DAYS_INSTALMENT of the oldest installment", "min"),
    FeatureSpec("late_count", "timeliness", "installments paid after the due date", "count", True),
    FeatureSpec("late_rate", "timeliness", "share of paid installments that were late", "share"),
    FeatureSpec("severe_late_rate", "timeliness", "share paid more than 30 days late", "share"),
    FeatureSpec(
        "days_late_mean", "timeliness", "mean (entry - due) days, negative = early", "mean"
    ),
    FeatureSpec("days_late_max", "timeliness", "max days late", "max"),
    FeatureSpec("days_late_min", "timeliness", "earliest payment (most negative days late)", "min"),
    FeatureSpec("days_late_std", "timeliness", "std of days late (timing variability)", "std"),
    FeatureSpec(
        "late_days_when_late_mean",
        "timeliness",
        "mean days late over late installments only",
        "mean",
    ),
    FeatureSpec(
        "unpaid_rate", "completeness", "share of installments with no payment date", "share"
    ),
    FeatureSpec(
        "payment_ratio_mean", "completeness", "mean paid / scheduled (clipped to [0, 2])", "mean"
    ),
    FeatureSpec("payment_ratio_std", "completeness", "std of paid / scheduled", "std"),
    FeatureSpec(
        "underpaid_rate", "completeness", "share of installments paid short by > 1 unit", "share"
    ),
    FeatureSpec(
        "overpaid_rate", "completeness", "share of installments paid over by > 1 unit", "share"
    ),
    FeatureSpec("scheduled_sum", "completeness", "total scheduled installment amount", "sum", True),
    FeatureSpec("paid_sum", "completeness", "total amount paid", "sum", True),
    FeatureSpec(
        "payment_gap_sum",
        "completeness",
        "sum of (scheduled - paid), positive = short-fall",
        "sum",
        True,
    ),
    FeatureSpec("paid_share", "completeness", "paid_sum / scheduled_sum", "ratio"),
    FeatureSpec(
        "n_installments_12m", "recent", "installments due in the last 365 days", "count", True
    ),
    FeatureSpec("late_rate_12m", "recent", "late share over the last 365 days", "share"),
    FeatureSpec("underpaid_rate_12m", "recent", "under-paid share over the last 365 days", "share"),
]


def installment_level(rows: pd.DataFrame) -> pd.DataFrame:
    """Sum payment parts to one row per installment and derive behaviour columns."""
    require_historical(rows, "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT")
    g = rows.groupby(INSTALLMENT_KEY, sort=False)
    inst = g.agg(
        SK_ID_CURR=("SK_ID_CURR", "first"),
        scheduled=("AMT_INSTALMENT", "max"),
        paid=("AMT_PAYMENT", "sum"),
        days_instalment=("DAYS_INSTALMENT", "max"),
        days_entry=("DAYS_ENTRY_PAYMENT", "max"),
        n_parts=("AMT_PAYMENT", "size"),
    ).reset_index()
    inst["days_late"] = (inst["days_entry"] - inst["days_instalment"]).astype(FLOAT)
    inst["is_paid"] = inst["days_entry"].notna()
    inst["is_late"] = (inst["days_late"] > 0).astype(FLOAT).where(inst["is_paid"])
    inst["is_severe_late"] = (
        (inst["days_late"] > SEVERE_LATE_DAYS).astype(FLOAT).where(inst["is_paid"])
    )
    inst["late_days_when_late"] = inst["days_late"].where(inst["days_late"] > 0)
    inst["payment_ratio"] = safe_ratio(inst["paid"], inst["scheduled"]).clip(upper=RATIO_CLIP)
    inst["is_underpaid"] = (inst["paid"] < inst["scheduled"] - AMOUNT_TOLERANCE).astype(FLOAT)
    inst["is_overpaid"] = (inst["paid"] > inst["scheduled"] + AMOUNT_TOLERANCE).astype(FLOAT)
    inst["is_unpaid"] = (~inst["is_paid"]).astype(FLOAT)
    inst["is_split"] = (inst["n_parts"] > 1).astype(FLOAT)
    inst["gap"] = (inst["scheduled"] - inst["paid"]).astype(FLOAT)
    recent = inst["days_instalment"] >= -RECENT_DAYS
    inst["is_recent"] = recent.astype(FLOAT)
    inst["is_late_recent"] = inst["is_late"].where(recent)
    inst["is_underpaid_recent"] = inst["is_underpaid"].where(recent)
    return inst


def build_installments_features(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per ``SK_ID_CURR`` from the raw payment rows."""
    inst = installment_level(rows)
    out = group_agg(
        inst,
        ID,
        {
            "n_installments": ("scheduled", "size"),
            "n_contracts": ("SK_ID_PREV", "nunique"),
            "n_split_payments": ("is_split", "sum"),
            "days_since_last_instalment": ("days_instalment", "max"),
            "history_span_days": ("days_instalment", "min"),
            "late_count": ("is_late", "sum"),
            "late_rate": ("is_late", "mean"),
            "severe_late_rate": ("is_severe_late", "mean"),
            "days_late_mean": ("days_late", "mean"),
            "days_late_max": ("days_late", "max"),
            "days_late_min": ("days_late", "min"),
            "days_late_std": ("days_late", "std"),
            "late_days_when_late_mean": ("late_days_when_late", "mean"),
            "unpaid_rate": ("is_unpaid", "mean"),
            "payment_ratio_mean": ("payment_ratio", "mean"),
            "payment_ratio_std": ("payment_ratio", "std"),
            "underpaid_rate": ("is_underpaid", "mean"),
            "overpaid_rate": ("is_overpaid", "mean"),
            "scheduled_sum": ("scheduled", "sum"),
            "paid_sum": ("paid", "sum"),
            "payment_gap_sum": ("gap", "sum"),
            "n_installments_12m": ("is_recent", "sum"),
            "late_rate_12m": ("is_late_recent", "mean"),
            "underpaid_rate_12m": ("is_underpaid_recent", "mean"),
        },
    )
    out["paid_share"] = safe_ratio(out["paid_sum"], out["scheduled_sum"])
    return finalize_customer_table(out, prefix=PREFIX, specs=SPECS)


def audit_installments(rows: pd.DataFrame) -> dict[str, Any]:
    late = rows["DAYS_ENTRY_PAYMENT"] - rows["DAYS_INSTALMENT"]
    ratio = rows["AMT_PAYMENT"] / rows["AMT_INSTALMENT"].where(rows["AMT_INSTALMENT"] > 0)
    dup_key = rows.duplicated(INSTALLMENT_KEY).sum()
    return {
        "grain": {
            "n_rows": int(len(rows)),
            "n_customers": int(rows[ID].nunique()),
            "n_previous_credits": int(rows["SK_ID_PREV"].nunique()),
            "rows_sharing_an_installment_key": int(dup_key),
            "share_sharing_an_installment_key": float(dup_key / max(len(rows), 1)),
        },
        "temporal": {
            c: {
                "min": float(rows[c].min()),
                "max": float(rows[c].max()),
                "n_positive": int((rows[c] > 0).sum()),
            }
            for c in ("DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT")
        },
        "days_late_convention": {
            "definition": "DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT; positive = paid after due date",
            "mean": float(late.mean()),
            "median": float(late.median()),
            "share_positive": float((late > 0).mean()),
            "share_over_30": float((late > SEVERE_LATE_DAYS).mean()),
            "max": float(late.max()),
        },
        "amounts": {c: audit_series(rows[c]) for c in ("AMT_INSTALMENT", "AMT_PAYMENT")},
        "payment_ratio_rows": {
            "median": float(ratio.median()),
            "share_below_0_99": float((ratio < 0.99).mean()),
            "share_above_1_01": float((ratio > 1.01).mean()),
            "n_zero_scheduled": int((rows["AMT_INSTALMENT"] == 0).sum()),
        },
        "anomalies": {
            "missing_payment_date_rows": int(rows["DAYS_ENTRY_PAYMENT"].isna().sum()),
            "missing_payment_amount_rows": int(rows["AMT_PAYMENT"].isna().sum()),
            "negative_amount_rows": int(
                ((rows["AMT_PAYMENT"] < 0) | (rows["AMT_INSTALMENT"] < 0)).sum()
            ),
        },
    }


def load_and_build(
    *, raw_dir=None, nrows: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    kwargs = {"nrows": nrows}
    if raw_dir is not None:
        kwargs["raw_dir"] = raw_dir
    rows = read_raw_table("installments", usecols=USECOLS, dtypes=DTYPES, **kwargs)
    audit = audit_installments(rows)
    features = build_installments_features(rows)
    sizes = {"installments_rows": int(len(rows))}
    del rows
    return features, audit, sizes


__all__ = [
    "SPECS",
    "audit_installments",
    "build_installments_features",
    "installment_level",
    "load_and_build",
]
