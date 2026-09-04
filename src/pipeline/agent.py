"""
Stage 4: Investigation & Attribution Agent — the single LLM touchpoint in
the pipeline (see plan.md "Design principle: LLM only where judgement is
genuinely required"). Everything upstream (stages 1-3) is deterministic
code; this stage reasons over the evidence pack those stages produced,
optionally drills down via read-only tools back into the same deterministic
functions, and emits one structured verdict. It never computes a number —
every figure it can cite comes from the evidence pack or a drill-down tool
result, never from its own arithmetic.

The client is injected (not constructed here) so tests can pass a fake
that mimics the same `.messages.create(...)` shape as `anthropic.Anthropic`
without hitting the real API — see tests/test_agent.py.
"""

from __future__ import annotations

import json

import pandas as pd

from .detect import _monthly_category_series, product_changes
from .changepoint import full_month_index

DEFAULT_MODEL = "claude-opus-5"
MAX_ITERATIONS = 6

SYSTEM_PROMPT = """You are the investigation and attribution agent in a revenue-leakage detection \
system for B2B retail accounts. You are the ONLY reasoning step in this pipeline — every stage \
before you (data cleaning, baseline computation, change-point detection) is deterministic code, \
and every number you see was computed there, not by you. You must never invent, adjust, or compute \
a number yourself; only cite facts that appear in the evidence pack or in a tool result.

Your job: given one account's evidence pack, decide whether it shows healthy behaviour, temporary \
variation, or structural revenue leakage — where the value is leaking from if so, and whether the \
evidence supports any confident call at all.

MOST ACCOUNTS THAT LOOK LIKE LEAKS ARE NOT LEAKS. In a realistic book, the majority of accounts \
showing a scary-looking recent number are seasonal, recovered, growing, inflated by a past bulk \
order, missing a month of data, or drifting within their normal band. Flagging everything is not \
caution — it is the most common failure mode, and it is wrong far more often than it is right. \
Equally, a genuinely leaking account can look perfectly healthy at the topline. Your value is the \
discrimination, not the alarm.

CHECK EVERY DIMENSION, NOT JUST REVENUE. `analysis_dimensions` tells you which of these the input \
file could actually support — if a dimension is false there, you did not measure it, and you must \
say so rather than imply it was fine:
- revenue_decline: graded material_decline / mild_drift / stable / growth. `mild_drift` is NOT a \
leak on its own; it is the normal band.
- margin: an account can hold revenue perfectly flat and still bleed value. `erosion_detected` at \
flat revenue is one of the most important findings you can make, and revenue-only reasoning misses \
it completely. Always look at margin before concluding an account is healthy.
- discount: `creep_detected` means the same goods are being sold at a steadily deeper discount. \
Volume, mix and order pattern can all look untouched while the money leaves through the price.
- tier_mix: `downgrade_detected` (value sliding out of High tier into Low) at flat revenue is the \
classic hidden leak. `premiumisation_detected` is the exact opposite and is GOOD NEWS.
- order_pattern: `fragmentation_detected` (more orders, each smaller) is an early multi-sourcing \
signal in its own right, even with no category change-point and near-flat revenue.
- category_changes: a category with `defected: true` has stopped completely and stayed stopped — \
name that category specifically.

DIRECTION IS NOT MAGNITUDE. Rising margin, rising high-tier share, and rising revenue are all \
positive. An account buying fewer units but trading UP to premium lines has flat revenue, higher \
margin and a higher high-tier share — that is a healthy account, and flagging it is a serious \
error. Never treat "the mix changed" as automatically bad; read which way it moved.

RULE OUT THE INNOCENT EXPLANATIONS BEFORE FLAGGING:
- `seasonality.status == "confirmed"` means this same dip happened in the same calendar window a \
year earlier (the echoing months are listed). That is strong evidence the recent dip is seasonal, \
not structural. Cite the prior-year months by name.
- `dip_episodes` lists every below-normal run in the history with whether it recovered. A dip that \
`recovered` months ago is a resolved incident, not a current leak. Only an `is_ongoing` episode or \
a non-recovered change-point is a live problem.
- `data_quality.gaps.months_with_no_orders`: a month with no orders is a hole in the data, not a \
month of zero trading. It drags trailing averages down by itself. Do not read it as a decline.
- `data_quality.outlier_months`: a one-off stock-up month inflates whatever window it lands in. The \
ordinary months after it are a return to normal, not a decline.
- `data_quality.returns`: credit notes are already netted into every figure. A return is not \
leakage.
- An account manager change (`account_manager_changed`) is a correlation. It is never, by itself, \
evidence of a cause. You may note it as context; do not attribute a leak to it.

ATTRIBUTION HONESTY. Name a category only when a category actually explains the loss. If the \
decline is broad-based across the whole book (no single category defected, no single category \
dominates the change), say that it is whole-account disengagement and leave attributed_categories \
EMPTY. Inventing a scapegoat category for a diffuse decline is worse than naming none: it sends \
the account team after the wrong thing. Likewise, if the leak is in margin or discount rather than \
in any category's volume, put the dimension in leak_dimensions and leave the category list empty \
unless a specific category is genuinely responsible.

MATERIALITY: a change-point in a category that was only a small share of this account's baseline \
revenue (see category_mix.baseline_share) is much weaker evidence than one in a major category, \
even if the percentage decline looks large — a low-revenue category is where random noise most \
easily crosses a threshold by chance. Do not build a confident verdict on a single low-share \
category; find corroborating evidence or lower your confidence.

SEASONALITY WITH SHORT HISTORY: `seasonal_precedent` on a category is deliberately strict — it \
needs 2+ prior occurrences of the same calendar month, which takes about three years, so it will \
often say "insufficient_history". That is not the same as "no seasonality". `prior_year_echo` and \
the account-level `seasonality` field are the checks that two years of history can support. If \
seasonality is still plausible and unresolved, call get_category_seasonal_breakdown or \
get_tier_monthly_series and judge the calendar history yourself.

DEFER WHEN THE EVIDENCE DOESN'T SUPPORT A CLEAN ANSWER. This is the most important instruction in \
this prompt. If history is too short (see data_sufficiency — flags like history_too_short_for_baseline \
or cannot_check_prior_year_seasonality), the shift could plausibly be seasonal but you can't \
confirm it, or the signals conflict, set defer=true, use low confidence, and name in \
data_needed_if_deferring exactly what data would resolve it. An account with only a few months of \
history genuinely cannot be classified temporary vs structural — no amount of reasoning fixes \
missing months, and guessing is the failure. A deferred, low-confidence, well-reasoned answer is \
scored HIGHER than a forced confident one — do not manufacture a verdict to sound decisive.

You may call the drill-down tools as many times as you need before answering. When you have enough \
evidence, call submit_verdict exactly once with your final structured answer. Every entry in \
cited_facts must reference a specific fact from the evidence pack or a tool result (e.g. \
"Diagnostic Equipment: defected, 10 consecutive months at zero since 2025-11, prior_year_echo \
not_present" or "margin 32.2% -> 19.9% (-12.3pp) with revenue flat at +2.5%") — not a vague \
restatement."""


def _tool_get_category_monthly_series(df: pd.DataFrame, account_id: str, category: str) -> dict:
    acc_df = df[df["account_id"] == account_id]
    months = full_month_index(acc_df)
    series = _monthly_category_series(acc_df, category, months)
    return {str(m): round(float(v), 2) for m, v in series.items()}


def _tool_get_category_seasonal_breakdown(df: pd.DataFrame, account_id: str, category: str) -> dict:
    acc_df = df[df["account_id"] == account_id]
    months = full_month_index(acc_df)
    series = _monthly_category_series(acc_df, category, months)
    by_calendar_month: dict[int, list] = {}
    for month, value in series.items():
        by_calendar_month.setdefault(month.month, []).append({"period": str(month), "revenue": round(float(value), 2)})
    return {str(k): v for k, v in sorted(by_calendar_month.items())}


def _tool_get_product_changes(df: pd.DataFrame, account_id: str, category: str | None = None) -> list:
    acc_df = df[df["account_id"] == account_id]
    changes = product_changes(acc_df, top_n=50)
    if category:
        changes = [c for c in changes if c["category"] == category]
    return changes


def _tool_get_monthly_economics(df: pd.DataFrame, account_id: str) -> dict:
    """Month-by-month revenue, margin, margin rate and average discount side
    by side — the view that separates a volume story from a price story."""
    acc_df = df[df["account_id"] == account_id].copy()
    months = full_month_index(acc_df)
    acc_df["month"] = acc_df["date"].dt.to_period("M")

    revenue = acc_df.groupby("month")["revenue"].sum().reindex(months, fill_value=0.0)
    has_margin = "margin" in acc_df.columns and acc_df["margin"].notna().any()
    margin = (
        acc_df.groupby("month")["margin"].sum().reindex(months, fill_value=0.0)
        if has_margin else None
    )

    discount = None
    if "discount_pct" in acc_df.columns and acc_df["discount_pct"].notna().any():
        sales = acc_df[acc_df.get("is_return", 0) != 1]
        weight = (
            (sales["list_price"] * sales["quantity"]).abs()
            if "list_price" in sales.columns and sales["list_price"].notna().any()
            else sales["revenue"].abs()
        )
        weighted = sales.assign(_w=weight, _wd=sales["discount_pct"] * weight)
        grouped = weighted.groupby("month")[["_w", "_wd"]].sum()
        discount = (grouped["_wd"] / grouped["_w"]).reindex(months)

    out = {}
    for month in months:
        row = {"revenue": round(float(revenue.loc[month]), 2)}
        if margin is not None:
            month_margin = float(margin.loc[month])
            row["margin"] = round(month_margin, 2)
            row["margin_pct"] = (
                round(month_margin / float(revenue.loc[month]), 4)
                if float(revenue.loc[month]) else None
            )
        if discount is not None:
            value = discount.loc[month]
            row["avg_discount_pct"] = None if pd.isna(value) else round(float(value), 4)
        out[str(month)] = row
    return out


def _tool_get_tier_monthly_series(df: pd.DataFrame, account_id: str) -> dict:
    """Monthly revenue split by product value tier, plus the High-tier share
    of each month — for judging a mix shift's direction directly."""
    acc_df = df[df["account_id"] == account_id].copy()
    if "tier" not in acc_df.columns or not acc_df["tier"].notna().any():
        return {"error": "This dataset has no product value-tier column; tier mix cannot be analysed."}

    months = full_month_index(acc_df)
    acc_df["month"] = acc_df["date"].dt.to_period("M")
    pivot = (
        acc_df.pivot_table(index="month", columns="tier", values="revenue", aggfunc="sum")
        .reindex(months)
        .fillna(0.0)
    )
    out = {}
    for month in months:
        row = {tier: round(float(pivot.loc[month, tier]), 2) for tier in pivot.columns}
        total = sum(row.values())
        row["high_tier_share"] = round(row.get("High", 0.0) / total, 4) if total else None
        out[str(month)] = row
    return out


DRILLDOWN_TOOLS = [
    {
        "name": "get_category_monthly_series",
        "description": "Raw monthly revenue for one category for this account, full history, no smoothing — for inspecting a change-point or trend in finer detail than the evidence pack's summary.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_category_seasonal_breakdown",
        "description": "This category's monthly revenue grouped by calendar month across all years in history — use this to judge seasonality yourself when the evidence pack's seasonal_precedent field says insufficient_history or not_present but seasonality still seems plausible; that field requires 2+ prior years of the same calendar month before it will confirm anything.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_product_changes",
        "description": "Product-level revenue changes (disappeared/declined/new) for this account, optionally filtered to one category — for locating which specific products, not just which category, are responsible.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": ["string", "null"]}},
            "required": ["category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_monthly_economics",
        "description": "Month-by-month revenue, margin, margin rate and average discount for this account, side by side. Use this to separate a volume story from a price story — e.g. to see whether flat revenue is being held up while margin rate falls, or to trace when a discount started creeping.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_tier_monthly_series",
        "description": "Monthly revenue split by product value tier (High/Mid/Low) with each month's High-tier share. Use this to judge the DIRECTION of a mix shift yourself — value moving out of High tier is a downgrade, value moving into it is premiumisation.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

SUBMIT_VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit your final, structured investigation result. Call this exactly once, when you have enough evidence to answer (including a deferred answer).",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["healthy", "leakage_detected", "insufficient_data"],
            },
            "temporary_or_structural": {
                "type": "string",
                "enum": ["temporary", "structural", "not_applicable"],
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "defer": {"type": "boolean"},
            "leak_dimensions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["revenue", "margin", "discount", "category_mix", "tier_mix",
                             "order_pattern"],
                },
                "description": "Which dimension(s) the value is actually leaking through. Empty if healthy or deferring. A margin or discount leak at flat revenue must say so here.",
            },
            "attributed_categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Categories responsible for the leakage. Empty if healthy, deferring, or if the decline is broad-based with no single category responsible — do not name a scapegoat category for a diffuse decline.",
            },
            "cited_facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific facts from the evidence pack or tool results that this verdict rests on.",
            },
            "narrative": {
                "type": "string",
                "description": "A short explanation a business stakeholder could act on directly.",
            },
            "recommended_actions": {"type": "array", "items": {"type": "string"}},
            "data_needed_if_deferring": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What additional data would resolve the ambiguity. Empty if not deferring.",
            },
        },
        "required": [
            "verdict", "temporary_or_structural", "confidence", "defer",
            "leak_dimensions", "attributed_categories", "cited_facts", "narrative",
            "recommended_actions", "data_needed_if_deferring",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}


def pack_for_prompt(evidence_pack: dict) -> dict:
    """The evidence pack as the model sees it: every underscore-prefixed key
    removed.

    `build_evidence_pack` carries a `_presentation` block of chart series and
    per-category monthly revenue for the PDF report. That data is far more
    verbose than anything Stage 4 needs, and the prompt is tuned against all
    18 reference accounts, so quietly growing the model's input is a
    regression the scripted-client tests cannot detect. Stripping it here
    makes the report and the prompt independently extensible.
    """
    return {k: v for k, v in evidence_pack.items() if not k.startswith("_")}


class AgentError(Exception):
    """Raised when the agent loop fails to reach a submit_verdict call
    (max iterations exhausted, or the model stopped for an unexpected
    reason) — surfaced as a clear error rather than a silent bad verdict."""


def _execute_tool(df: pd.DataFrame, account_id: str, name: str, tool_input: dict):
    if name == "get_category_monthly_series":
        return _tool_get_category_monthly_series(df, account_id, tool_input["category"])
    if name == "get_category_seasonal_breakdown":
        return _tool_get_category_seasonal_breakdown(df, account_id, tool_input["category"])
    if name == "get_product_changes":
        return _tool_get_product_changes(df, account_id, tool_input.get("category"))
    if name == "get_monthly_economics":
        return _tool_get_monthly_economics(df, account_id)
    if name == "get_tier_monthly_series":
        return _tool_get_tier_monthly_series(df, account_id)
    raise AgentError(f"Unknown tool requested by model: {name}")


def investigate(
    client,
    df: pd.DataFrame,
    account_id: str,
    evidence_pack: dict,
    model: str = DEFAULT_MODEL,
    max_iterations: int = MAX_ITERATIONS,
) -> dict:
    """Run the Stage 4 agent loop for one account. Returns the parsed
    submit_verdict input dict. Raises AgentError if the model never
    submits a verdict within max_iterations."""
    tools = DRILLDOWN_TOOLS + [SUBMIT_VERDICT_TOOL]
    messages = [{"role": "user", "content": json.dumps(pack_for_prompt(evidence_pack))}]

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        submit_block = next((b for b in tool_use_blocks if b.name == "submit_verdict"), None)
        if submit_block is not None:
            return submit_block.input

        if not tool_use_blocks:
            raise AgentError(
                f"Model stopped (stop_reason={response.stop_reason!r}) without calling submit_verdict."
            )

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_use_blocks:
            try:
                result = _execute_tool(df, account_id, block.name, block.input)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result),
                })
            except Exception as e:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": f"Error: {e}", "is_error": True,
                })
        messages.append({"role": "user", "content": tool_results})

    raise AgentError(f"Exceeded max_iterations={max_iterations} without a submit_verdict call.")
