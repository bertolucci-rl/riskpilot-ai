"""Customer-level features from ``previous_application.csv``.

Grain (verified): one row per previous Home Credit application ``SK_ID_PREV``
(unique), many per ``SK_ID_CURR``.

Temporal semantics: ``DAYS_DECISION`` ("relative to current application when
was the decision made") is in [-2922, -1] for every row, so every previous
application was decided before the current one. The five other ``DAYS_*``
columns (``DAYS_FIRST_DRAWING``, ``DAYS_FIRST_DUE``, ``DAYS_LAST_DUE_1ST_VERSION``,
``DAYS_LAST_DUE``, ``DAYS_TERMINATION``) mix a 365243 placeholder with genuine
positive values (schedules that end after the current application) and are
**excluded**: their sign is ambiguous and part of their information describes
the future relative to the decision. Amounts, statuses, reject reasons and
product attributes are properties of the past application itself.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from riskpilot.features.relational.common import (
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
from riskpilot.features.relational.temporal import require_historical

SOURCE = "previous"
PREFIX = "previous"
LEAKAGE = (
    "safe: only DAYS_DECISION (always < 0) is used as a time anchor; the ambiguous "
    "DAYS_FIRST_DRAWING/FIRST_DUE/LAST_DUE*/TERMINATION columns are excluded"
)

USECOLS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "NAME_CONTRACT_TYPE",
    "AMT_ANNUITY",
    "AMT_APPLICATION",
    "AMT_CREDIT",
    "AMT_DOWN_PAYMENT",
    "AMT_GOODS_PRICE",
    "RATE_DOWN_PAYMENT",
    "NAME_CONTRACT_STATUS",
    "DAYS_DECISION",
    "CODE_REJECT_REASON",
    "NAME_CLIENT_TYPE",
    "NAME_PORTFOLIO",
    "NAME_PRODUCT_TYPE",
    "CNT_PAYMENT",
    "NAME_YIELD_GROUP",
    "NFLAG_INSURED_ON_APPROVAL",
]
DTYPES = {
    "SK_ID_PREV": "int32",
    "SK_ID_CURR": "int32",
    "NAME_CONTRACT_TYPE": "category",
    "AMT_ANNUITY": "float32",
    "AMT_APPLICATION": "float32",
    "AMT_CREDIT": "float32",
    "AMT_DOWN_PAYMENT": "float32",
    "AMT_GOODS_PRICE": "float32",
    "RATE_DOWN_PAYMENT": "float32",
    "NAME_CONTRACT_STATUS": "category",
    "DAYS_DECISION": "float32",
    "CODE_REJECT_REASON": "category",
    "NAME_CLIENT_TYPE": "category",
    "NAME_PORTFOLIO": "category",
    "NAME_PRODUCT_TYPE": "category",
    "CNT_PAYMENT": "float32",
    "NAME_YIELD_GROUP": "category",
    "NFLAG_INSURED_ON_APPROVAL": "float32",
}
EXCLUDED_TEMPORAL_COLUMNS = (
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
)
RECENT_DAYS = 365

SPECS: list[FeatureSpec] = [
    FeatureSpec("count", "counts", "number of previous applications", "count", True),
    FeatureSpec("approved_count", "counts", "applications with status Approved", "count", True),
    FeatureSpec("refused_count", "counts", "applications with status Refused", "count", True),
    FeatureSpec("canceled_count", "counts", "applications with status Canceled", "count", True),
    FeatureSpec(
        "unused_offer_count", "counts", "applications with status Unused offer", "count", True
    ),
    FeatureSpec("approval_rate", "rates", "approved / all previous applications", "ratio"),
    FeatureSpec("refusal_rate", "rates", "refused / all previous applications", "ratio"),
    FeatureSpec("cancellation_rate", "rates", "canceled / all previous applications", "ratio"),
    FeatureSpec(
        "last_application_refused",
        "rates",
        "most recent previous application (by DAYS_DECISION) was refused",
        "last",
    ),
    FeatureSpec(
        "last_application_approved",
        "rates",
        "most recent previous application was approved",
        "last",
    ),
    FeatureSpec(
        "decisions_last_year",
        "recency",
        "previous applications decided in the last 365 days",
        "count",
        True,
    ),
    FeatureSpec(
        "days_decision_max", "recency", "DAYS_DECISION of the most recent application", "max"
    ),
    FeatureSpec("days_decision_min", "recency", "DAYS_DECISION of the oldest application", "min"),
    FeatureSpec("days_decision_mean", "recency", "mean DAYS_DECISION", "mean"),
    FeatureSpec(
        "amt_application_mean", "amounts", "mean requested amount (AMT_APPLICATION > 0)", "mean"
    ),
    FeatureSpec("amt_application_max", "amounts", "max requested amount", "max"),
    FeatureSpec(
        "amt_credit_approved_mean",
        "amounts",
        "mean granted AMT_CREDIT over approved applications",
        "mean",
    ),
    FeatureSpec(
        "amt_credit_approved_max",
        "amounts",
        "max granted AMT_CREDIT over approved applications",
        "max",
    ),
    FeatureSpec(
        "amt_credit_approved_sum",
        "amounts",
        "sum of granted AMT_CREDIT over approved applications",
        "sum",
        True,
    ),
    FeatureSpec(
        "amt_annuity_approved_mean", "amounts", "mean annuity over approved applications", "mean"
    ),
    FeatureSpec("amt_goods_price_mean", "amounts", "mean goods price", "mean"),
    FeatureSpec("amt_down_payment_mean", "amounts", "mean down payment", "mean"),
    FeatureSpec("rate_down_payment_mean", "amounts", "mean normalized down-payment rate", "mean"),
    FeatureSpec(
        "credit_to_application_ratio_mean",
        "amounts",
        "mean AMT_CREDIT / AMT_APPLICATION (application > 0)",
        "mean",
    ),
    FeatureSpec(
        "cnt_payment_approved_mean", "terms", "mean term (CNT_PAYMENT) of approved credits", "mean"
    ),
    FeatureSpec("cnt_payment_approved_max", "terms", "max term of approved credits", "max"),
    FeatureSpec("cash_loan_share", "products", "share of Cash loans", "share"),
    FeatureSpec("consumer_loan_share", "products", "share of Consumer loans", "share"),
    FeatureSpec("revolving_loan_share", "products", "share of Revolving loans", "share"),
    FeatureSpec("portfolio_pos_share", "products", "share of POS portfolio", "share"),
    FeatureSpec("portfolio_cash_share", "products", "share of Cash portfolio", "share"),
    FeatureSpec("portfolio_cards_share", "products", "share of Cards portfolio", "share"),
    FeatureSpec("x_sell_share", "products", "share of x-sell product type", "share"),
    FeatureSpec("walk_in_share", "products", "share of walk-in product type", "share"),
    FeatureSpec("yield_high_share", "products", "share of high yield group", "share"),
    FeatureSpec(
        "yield_low_share", "products", "share of low_normal / low_action yield groups", "share"
    ),
    FeatureSpec(
        "client_new_share", "products", "share of applications made as a New client", "share"
    ),
    FeatureSpec("insured_rate", "products", "mean NFLAG_INSURED_ON_APPROVAL", "mean"),
    FeatureSpec("reject_hc_count", "rejections", "refusals with reason HC", "count", True),
    FeatureSpec("reject_limit_count", "rejections", "refusals with reason LIMIT", "count", True),
    FeatureSpec(
        "reject_scoring_count", "rejections", "refusals with reason SCO or SCOFR", "count", True
    ),
]


def build_previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    """One row per ``SK_ID_CURR`` from ``previous_application``."""
    require_historical(prev, "DAYS_DECISION")
    p = prev.copy(deep=False)
    status = p["NAME_CONTRACT_STATUS"].astype(str)
    p["approved"] = (status == "Approved").astype("int8")
    p["refused"] = (status == "Refused").astype("int8")
    p["canceled"] = (status == "Canceled").astype("int8")
    p["unused"] = (status == "Unused offer").astype("int8")
    p["recent"] = (p["DAYS_DECISION"] >= -RECENT_DAYS).astype("int8")
    p["amt_application_pos"] = p["AMT_APPLICATION"].where(p["AMT_APPLICATION"] > 0)
    p["amt_credit_approved"] = p["AMT_CREDIT"].where(p["approved"] == 1)
    p["amt_annuity_approved"] = p["AMT_ANNUITY"].where(p["approved"] == 1)
    p["cnt_payment_approved"] = p["CNT_PAYMENT"].where(p["approved"] == 1)
    p["credit_to_application"] = safe_ratio(p["AMT_CREDIT"], p["AMT_APPLICATION"])
    ctype = p["NAME_CONTRACT_TYPE"].astype(str)
    p["cash_loan"] = (ctype == "Cash loans").astype("int8")
    p["consumer_loan"] = (ctype == "Consumer loans").astype("int8")
    p["revolving_loan"] = (ctype == "Revolving loans").astype("int8")
    portfolio = p["NAME_PORTFOLIO"].astype(str)
    p["portfolio_pos"] = (portfolio == "POS").astype("int8")
    p["portfolio_cash"] = (portfolio == "Cash").astype("int8")
    p["portfolio_cards"] = (portfolio == "Cards").astype("int8")
    product = p["NAME_PRODUCT_TYPE"].astype(str)
    p["x_sell"] = (product == "x-sell").astype("int8")
    p["walk_in"] = (product == "walk-in").astype("int8")
    yield_group = p["NAME_YIELD_GROUP"].astype(str)
    p["yield_high"] = (yield_group == "high").astype("int8")
    p["yield_low"] = yield_group.isin(["low_normal", "low_action"]).astype("int8")
    p["client_new"] = (p["NAME_CLIENT_TYPE"].astype(str) == "New").astype("int8")
    reason = p["CODE_REJECT_REASON"].astype(str)
    p["reject_hc"] = ((p["refused"] == 1) & (reason == "HC")).astype("int8")
    p["reject_limit"] = ((p["refused"] == 1) & (reason == "LIMIT")).astype("int8")
    p["reject_scoring"] = ((p["refused"] == 1) & reason.isin(["SCO", "SCOFR"])).astype("int8")

    out = group_agg(
        p,
        ID,
        {
            "count": ("SK_ID_PREV", "size"),
            "approved_count": ("approved", "sum"),
            "refused_count": ("refused", "sum"),
            "canceled_count": ("canceled", "sum"),
            "unused_offer_count": ("unused", "sum"),
            "decisions_last_year": ("recent", "sum"),
            "days_decision_max": ("DAYS_DECISION", "max"),
            "days_decision_min": ("DAYS_DECISION", "min"),
            "days_decision_mean": ("DAYS_DECISION", "mean"),
            "amt_application_mean": ("amt_application_pos", "mean"),
            "amt_application_max": ("AMT_APPLICATION", "max"),
            "amt_credit_approved_mean": ("amt_credit_approved", "mean"),
            "amt_credit_approved_max": ("amt_credit_approved", "max"),
            "amt_credit_approved_sum": ("amt_credit_approved", "sum"),
            "amt_annuity_approved_mean": ("amt_annuity_approved", "mean"),
            "amt_goods_price_mean": ("AMT_GOODS_PRICE", "mean"),
            "amt_down_payment_mean": ("AMT_DOWN_PAYMENT", "mean"),
            "rate_down_payment_mean": ("RATE_DOWN_PAYMENT", "mean"),
            "credit_to_application_ratio_mean": ("credit_to_application", "mean"),
            "cnt_payment_approved_mean": ("cnt_payment_approved", "mean"),
            "cnt_payment_approved_max": ("cnt_payment_approved", "max"),
            "cash_loan_share": ("cash_loan", "mean"),
            "consumer_loan_share": ("consumer_loan", "mean"),
            "revolving_loan_share": ("revolving_loan", "mean"),
            "portfolio_pos_share": ("portfolio_pos", "mean"),
            "portfolio_cash_share": ("portfolio_cash", "mean"),
            "portfolio_cards_share": ("portfolio_cards", "mean"),
            "x_sell_share": ("x_sell", "mean"),
            "walk_in_share": ("walk_in", "mean"),
            "yield_high_share": ("yield_high", "mean"),
            "yield_low_share": ("yield_low", "mean"),
            "client_new_share": ("client_new", "mean"),
            "insured_rate": ("NFLAG_INSURED_ON_APPROVAL", "mean"),
            "reject_hc_count": ("reject_hc", "sum"),
            "reject_limit_count": ("reject_limit", "sum"),
            "reject_scoring_count": ("reject_scoring", "sum"),
        },
    )
    out["approval_rate"] = safe_ratio(out["approved_count"], out["count"])
    out["refusal_rate"] = safe_ratio(out["refused_count"], out["count"])
    out["cancellation_rate"] = safe_ratio(out["canceled_count"], out["count"])
    latest = latest_per_group(p[[ID, "DAYS_DECISION", "refused", "approved"]], ID, "DAYS_DECISION")
    latest = latest.set_index(ID)
    out["last_application_refused"] = latest["refused"].reindex(out.index).astype("float32")
    out["last_application_approved"] = latest["approved"].reindex(out.index).astype("float32")
    return finalize_customer_table(out, prefix=PREFIX, specs=SPECS)


def audit_previous(
    prev: pd.DataFrame, temporal_extra: pd.DataFrame | None = None
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "grain": {
            "n_rows": int(len(prev)),
            "n_customers": int(prev[ID].nunique()),
            "sk_id_prev_unique": bool(prev["SK_ID_PREV"].is_unique),
        },
        "temporal": {
            "DAYS_DECISION": {
                "min": float(prev["DAYS_DECISION"].min()),
                "max": float(prev["DAYS_DECISION"].max()),
                "n_positive": int((prev["DAYS_DECISION"] > 0).sum()),
            }
        },
        "excluded_temporal_columns": list(EXCLUDED_TEMPORAL_COLUMNS),
        "contract_status": value_counts_dict(prev["NAME_CONTRACT_STATUS"]),
        "reject_reason": value_counts_dict(prev["CODE_REJECT_REASON"]),
        "contract_type": value_counts_dict(prev["NAME_CONTRACT_TYPE"]),
        "amounts": {
            c: audit_series(prev[c])
            for c in (
                "AMT_APPLICATION",
                "AMT_CREDIT",
                "AMT_ANNUITY",
                "AMT_DOWN_PAYMENT",
                "CNT_PAYMENT",
            )
        },
        "anomalies": {
            "zero_application_amount_rows": int((prev["AMT_APPLICATION"] == 0).sum()),
            "approved_with_zero_credit_rows": int(
                (
                    (prev["NAME_CONTRACT_STATUS"].astype(str) == "Approved")
                    & (prev["AMT_CREDIT"] == 0)
                ).sum()
            ),
            "negative_down_payment_rows": int((prev["AMT_DOWN_PAYMENT"] < 0).sum()),
        },
    }
    if temporal_extra is not None:
        audit["excluded_temporal_detail"] = {
            c: {
                "n_missing": int(temporal_extra[c].isna().sum()),
                "n_placeholder_365243": int((temporal_extra[c] == 365243).sum()),
                "n_positive_real": int(
                    ((temporal_extra[c] > 0) & (temporal_extra[c] != 365243)).sum()
                ),
                "min": float(temporal_extra[c].min()),
            }
            for c in EXCLUDED_TEMPORAL_COLUMNS
        }
    return audit


def load_and_build(
    *, raw_dir=None, nrows: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    kwargs = {"nrows": nrows}
    if raw_dir is not None:
        kwargs["raw_dir"] = raw_dir
    prev = read_raw_table("previous", usecols=USECOLS, dtypes=DTYPES, **kwargs)
    extra = read_raw_table(
        "previous",
        usecols=list(EXCLUDED_TEMPORAL_COLUMNS),
        dtypes=dict.fromkeys(EXCLUDED_TEMPORAL_COLUMNS, "float32"),
        **kwargs,
    )
    audit = audit_previous(prev, extra)
    del extra
    features = build_previous_features(prev)
    sizes = {"previous_rows": int(len(prev))}
    del prev
    return features, audit, sizes
