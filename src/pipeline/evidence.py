"""
Evidence pack assembly — combines Stage 1 (sufficiency), Stage 2 (baseline)
and Stage 3 (change detection) into the single structured JSON object the
Stage 4 LLM reasons over. The LLM never sees raw transactions and never
computes a number; every figure here traces back to deterministic code.

The pack is organised by DIMENSION (revenue, margin, discount, category mix,
tier mix, order pattern) rather than as a flat bag of numbers, because the
central discrimination this dataset demands is *which* dimension moved: flat
revenue with collapsing margin is a leak, flat revenue with rising margin is
a healthy account trading up, and the two are identical if you only look at
revenue. `analysis_dimensions` states which of those dimensions the input
file could actually support, so "not measured" is never mistaken for "no
change found".
"""

from __future__ import annotations

import pandas as pd

from .baseline import build_baseline
from .changepoint import full_month_index
from .detect import (
    _monthly_category_series,
    category_changes,
    data_gaps,
    dip_episodes,
    discount_creep,
    margin_erosion,
    order_pattern_shift,
    outlier_months,
    prior_year_echo,
    product_changes,
    returns_summary,
    revenue_decline,
    revenue_trend,
    tier_shift,
)
from .ingest import account_sufficiency, analysis_dimensions


def _monthly_revenue_series(monthly_series: dict) -> pd.Series:
    return pd.Series(
        {pd.Period(month, freq="M"): float(value) for month, value in monthly_series.items()}
    ).sort_index()


def _manager_tenures(acc_df: pd.DataFrame) -> list[dict]:
    """First month each account manager appears on this account's orders.

    A mid-history rep change is reported by identity as a bare boolean, which
    is enough to caveat a verdict but not enough to place the handover on a
    timeline next to the change it is often wrongly blamed for. Dating it
    lets the report say "rep changed in 2025-11, five months before the
    decline" — still a correlation, but one the reader can now judge.
    """
    if "account_manager" not in acc_df.columns:
        return []
    d = acc_df[["date", "account_manager"]].dropna()
    if d.empty:
        return []
    first_month = d.assign(month=d["date"].dt.to_period("M")).groupby("account_manager")["month"].min()
    return [
        {"account_manager": str(manager), "from_month": str(month)}
        for manager, month in first_month.sort_values().items()
    ]


def build_evidence_pack(df: pd.DataFrame, account_id: str) -> dict:
    acc_df = df[df["account_id"] == account_id].reset_index(drop=True)
    if acc_df.empty:
        raise ValueError(f"No rows for account_id={account_id!r}")

    sufficiency = account_sufficiency(df).get(account_id)
    baseline = build_baseline(df, account_id)
    changes = category_changes(acc_df, baseline["category_mix"]["high_value_categories"])
    products = product_changes(acc_df)
    trend = revenue_trend(baseline["overall_revenue"]["monthly_series"])
    revenue_series = _monthly_revenue_series(baseline["overall_revenue"]["monthly_series"])

    identity = {"account_id": account_id}
    for column, key in (("account_name", "account_name"), ("region", "region"),
                        ("account_manager", "account_manager")):
        if column in acc_df.columns and acc_df[column].notna().any():
            values = sorted(acc_df[column].dropna().astype(str).unique())
            identity[key] = values[0] if len(values) == 1 else values
            if column == "account_manager" and len(values) > 1:
                # A mid-history rep change correlates with all sorts of
                # things and causes none of them by itself. Surfaced as a
                # fact, explicitly labelled, so it can be mentioned without
                # being promoted to a cause.
                identity["account_manager_changed"] = True

    return {
        **identity,
        "data_sufficiency": sufficiency,
        "analysis_dimensions": analysis_dimensions(acc_df),
        "history": {
            "start_date": str(acc_df["date"].min().date()),
            "end_date": str(acc_df["date"].max().date()),
            "months_of_history": sufficiency["history_months"] if sufficiency else None,
            "order_count": sufficiency["order_count"] if sufficiency else None,
            "category_count": sufficiency["category_count"] if sufficiency else None,
        },
        "overall_revenue": baseline["overall_revenue"],
        "revenue_trend": trend,
        "revenue_decline": revenue_decline(trend),
        # Seasonality and dip history are checked on TOTAL revenue as well as
        # per category: a genuinely seasonal account dips across its whole
        # book at once, which no single category's change-point would show.
        "seasonality": prior_year_echo(revenue_series),
        "dip_episodes": dip_episodes(revenue_series),
        "margin": margin_erosion(baseline["margin_profile"]),
        "margin_profile": baseline["margin_profile"],
        "discount": discount_creep(baseline["discount_profile"]),
        "tier_mix": tier_shift(baseline["tier_mix"]),
        "category_mix": baseline["category_mix"],
        "category_changes": changes,
        "product_changes": products,
        "order_pattern": order_pattern_shift(baseline["order_behavior"]),
        "order_behavior": baseline["order_behavior"],
        "data_quality": {
            "gaps": data_gaps(acc_df),
            "outlier_months": outlier_months(acc_df),
            "returns": returns_summary(acc_df),
        },
        # Underscore-prefixed keys are for OUTPUT SURFACES ONLY (the PDF
        # report's tables and timeline) and are stripped before the pack is
        # serialized into the Stage 4 prompt — see agent.pack_for_prompt.
        #
        # The boundary exists so the report can be enriched without ever
        # changing what the model reads. Stage 4's prompt is tuned against
        # all 18 reference accounts and the unit tests use scripted verdicts,
        # so they cannot catch a regression caused by feeding the model more
        # text; keeping this channel out of the prompt makes that class of
        # regression impossible rather than merely unlikely.
        "_presentation": {
            "discount_profile": baseline["discount_profile"],
            "tier_mix_profile": baseline["tier_mix"],
            "category_monthly_revenue": _category_monthly_revenue(acc_df),
            "account_manager_tenures": _manager_tenures(acc_df),
        },
    }


def _category_monthly_revenue(acc_df: pd.DataFrame) -> dict:
    """Monthly revenue per category, full history.

    Only ever reaches the report, never the model — which is why it can
    afford to be this verbose. It dates a defection precisely ("stopped
    after 2025-10") instead of leaving the reader to subtract
    `consecutive_months_at_zero` from the end of history.
    """
    months = full_month_index(acc_df)
    return {
        str(category): {
            str(month): round(float(value), 2)
            for month, value in _monthly_category_series(acc_df, str(category), months).items()
        }
        for category in sorted(acc_df["category"].unique())
    }
