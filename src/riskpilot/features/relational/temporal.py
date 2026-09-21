"""Application-time eligibility; positive contractual maturity is not an event."""

from __future__ import annotations

import pandas as pd

POLICY_VERSION = "application-time-v2"
BUREAU_ORIGINATION_FIELDS = {"SK_ID_CURR", "SK_ID_BUREAU", "DAYS_CREDIT", "CREDIT_TYPE"}


def require_historical(frame: pd.DataFrame, *anchors: str) -> None:
    """Fail closed on new temporal anomalies instead of inventing a correction."""
    if "TARGET" in frame:
        raise ValueError("TARGET must never enter relational feature generation.")
    for anchor in anchors:
        if frame[anchor].gt(0).any():
            raise ValueError(f"Post-application observations in {anchor}; audit before building.")


def historical_rows(frame: pd.DataFrame, anchor: str) -> pd.DataFrame:
    """Require an observed historical anchor, including day/month zero."""
    if "TARGET" in frame:
        raise ValueError("TARGET must never enter relational feature generation.")
    return frame.loc[frame[anchor].le(0)].copy()


def sanitize_bureau_temporal_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep dated origination facts, mask snapshots unavailable at application.

    Loan existence, origination date and original credit type are historical facts.
    A later/undated update cannot establish earlier amounts, status or revised
    contractual terms. Monthly observations have their own eligibility check.
    Missing closure dates are normal for open loans; future realized closures
    contradict an application-time snapshot. Future *planned* maturity does not.
    """
    out = historical_rows(frame, "DAYS_CREDIT")
    unsafe = ~out["DAYS_CREDIT_UPDATE"].le(0) | out["DAYS_ENDDATE_FACT"].gt(0)
    for column in out.columns.difference(list(BUREAU_ORIGINATION_FIELDS)):
        out.loc[unsafe, column] = float("nan")
    return out
