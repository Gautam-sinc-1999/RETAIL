"""
Evidence timeline — turns the evidence pack into a dated, ordered log of
what actually changed on this account and how each fact bears on the
verdict. Deterministic; no LLM involvement, no new numbers.

Why this exists: the evidence pack is organised by DIMENSION, which is the
right shape for reasoning and the wrong shape for reading. A stakeholder
asked to accept "structural leakage, high confidence" wants the case laid
out the way a case is normally made — this happened, then this happened,
and here is what we checked and ruled out. Every event below is a
restatement of a fact Stages 1-3 already computed, re-keyed by date.

The `reads_as` field is the part that makes it an argument rather than a
list. Four values:

  concern     evidence that supports a leakage finding
  reassuring  evidence that argues against one
  checked     an innocent explanation that was explicitly tested
  context     scene-setting that is neither

A report that shows only the concerns is a prosecution, not an
investigation. `reassuring` and `checked` events are emitted even when the
verdict is leakage — they are what demonstrates the alternatives were
considered, and on a NO FLAG account they carry the entire explanation.
"""

from __future__ import annotations

from .changepoint import RECENT_MONTHS

CONCERN = "concern"
REASSURING = "reassuring"
CHECKED = "checked"
CONTEXT = "context"

# Reads_as values in the order a reader should meet them when the timeline
# is grouped rather than sorted by date.
READS_AS_ORDER = [CONCERN, REASSURING, CHECKED, CONTEXT]


def _event(month, headline, detail, dimension, reads_as, when=None, method=None,
           notable=None) -> dict:
    """One dated fact.

    `detail` is the business statement — what moved, by how much, and what it
    means for the account. `method` is how the call was made (the threshold
    it was judged against, the rule that classified it). They are separate
    fields because they serve different readers: an account owner acting on
    the finding needs the first, and a reviewer challenging it needs the
    second. Collapsing them buries a Rs 23k/month problem inside a sentence
    about percentage-point thresholds.
    """
    return {
        "month": month,
        "when": when or (month if month else "whole history"),
        "headline": headline,
        "detail": detail,
        "method": method,
        "dimension": dimension,
        "reads_as": reads_as,
        # Did something actually MOVE on this account? Distinct from whether
        # the movement is bad. A short report shows what moved in either
        # direction — an account trading up is a finding, and burying it with
        # the "nothing changed" rows loses the one thing worth telling the
        # account owner. Every concern is notable by definition; a reassuring
        # event is notable only when it reports a change rather than the
        # absence of one.
        "notable": bool(reads_as == CONCERN if notable is None else notable),
    }


def _rupees(value) -> str:
    if value is None:
        return "n/a"
    return f"Rs {value:,.0f}"


def _pct(value, digits: int = 1) -> str:
    """A ratio (0.42) rendered as a percentage. None is stated, never zero."""
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _pp(value, digits: int = 1) -> str:
    """An already-in-percentage-points delta, always signed."""
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}pp"


def _months(pack: dict) -> list[str]:
    return sorted((pack.get("overall_revenue") or {}).get("monthly_series") or {})


def _recent_window(pack: dict) -> tuple[str | None, str | None]:
    """First and last month of the trailing comparison window.

    Several detectors report a baseline-vs-recent delta with no date of its
    own. Anchoring those to the window they were actually measured over is
    what lets them sit on a timeline honestly instead of floating.
    """
    months = _months(pack)
    if not months:
        return None, None
    window = months[-RECENT_MONTHS:] if len(months) > RECENT_MONTHS else months
    return window[0], window[-1]


def _first_sustained(series: dict, predicate, min_run: int = 3) -> str | None:
    """First month beginning a run of >= min_run consecutive months that
    satisfy `predicate`, or that runs to the end of history.

    A single month across a threshold is noise on an account placing two or
    three orders a month; a run is a change. Returning the run's FIRST month
    is what makes the event datable — "margin first fell below its baseline
    band in 2025-11 and stayed there" rather than a bare delta.
    """
    months = sorted(series)
    run_start = None
    run_length = 0
    for month in months:
        value = series.get(month)
        if value is not None and predicate(value):
            if run_start is None:
                run_start = month
            run_length += 1
        else:
            if run_start is not None and run_length >= min_run:
                return run_start
            run_start, run_length = None, 0
    if run_start is not None and run_length >= min_run:
        return run_start
    return None


# --- individual event builders ---------------------------------------------


def _history_events(pack: dict) -> list[dict]:
    history = pack.get("history") or {}
    sufficiency = pack.get("data_sufficiency") or {}
    start = history.get("start_date")
    month = start[:7] if start else None

    detail = (
        f"{history.get('months_of_history')} months of history, "
        f"{history.get('order_count')} orders across {history.get('category_count')} categories. "
        f"Data sufficiency: {sufficiency.get('label', 'unknown')} (score {sufficiency.get('score')})."
    )
    events = [_event(month, "History begins", detail, "history", CONTEXT)]

    flags = sufficiency.get("flags") or []
    if flags:
        # Short history is not a neutral fact — it is the reason a confident
        # verdict may be unavailable, so it is raised as a concern about the
        # ANSWER rather than about the account.
        reads_as = CONCERN if sufficiency.get("label") != "sufficient" else CONTEXT
        events.append(_event(
            month,
            "Limits on what this history can establish",
            "Sufficiency flags: " + ", ".join(f.replace("_", " ") for f in flags) + ".",
            "history",
            reads_as,
        ))

    tenures = (pack.get("_presentation") or {}).get("account_manager_tenures") or []
    for tenure in (tenures if pack.get("account_manager_changed") else []):
        events.append(_event(
            tenure["from_month"],
            f"Account manager: {tenure['account_manager']}",
            "A rep change correlates with many things and causes none of them by itself. "
            "Recorded so it can be weighed against the dates below, not treated as a cause.",
            "data_quality",
            CHECKED,
        ))
    return events


def _revenue_events(pack: dict) -> list[dict]:
    events = []
    overall = pack.get("overall_revenue") or {}
    decline = pack.get("revenue_decline") or {}
    trend = pack.get("revenue_trend") or {}
    window_start, window_end = _recent_window(pack)
    window_label = f"{window_start} - {window_end}" if window_start else "recent window"

    change_point = overall.get("change_point")
    if change_point:
        recovered = change_point.get("recovered")
        events.append(_event(
            change_point["change_point_month"],
            f"Total revenue steps down {_pct(change_point.get('pct_decline'))}",
            f"Monthly revenue {_rupees(change_point.get('before_monthly_median'))} -> "
            f"{_rupees(change_point.get('after_monthly_median'))}; "
            f"sustained {change_point.get('sustained_months_since_change_point')} months since. "
            + ("Revenue has since returned to its earlier level." if recovered
               else "Revenue has not returned to its earlier level."),
            "revenue",
            REASSURING if recovered else CONCERN,
        ))

    status = decline.get("status")
    halves = decline.get("first_half_vs_second_half_pct_change")
    slope = decline.get("slope_pct_per_month")
    shared = f"First half vs second half {_pct(halves)}, trend {_pct(slope, 2)}/month."
    how = (f"Material decline requires both <= {_pct(-0.15)} half-over-half and "
           f"<= {_pct(-0.015, 2)}/month trend.")

    if status == "material_decline":
        events.append(_event(window_start, "Revenue is in material decline", shared,
                             "revenue", CONCERN, when=window_label, method=how))
    elif status == "mild_drift":
        events.append(_event(window_start, "Revenue drifting within its normal band",
                             shared + " This is drift, not a material decline.",
                             "revenue", CONTEXT, when=window_label, method=how))
    elif status == "growth":
        events.append(_event(window_start, "Revenue is growing", shared,
                             "revenue", REASSURING, when=window_label, method=how,
                             notable=True))
    elif status == "stable":
        events.append(_event(
            window_start, "Revenue is stable",
            shared + " A stable topline does not by itself mean a healthy account — "
            "the margin, discount and mix rows are what test that.",
            "revenue", REASSURING, when=window_label, method=how,
        ))

    cv = overall.get("monthly_revenue_coefficient_of_variation")
    if cv is not None and cv >= 0.5:
        events.append(_event(
            None, "This account is naturally volatile",
            f"Month-to-month revenue varies with a coefficient of variation of {cv:.2f}. "
            "Large single-month swings are normal here and are weak evidence on their own.",
            "revenue", CHECKED,
        ))

    direction = trend.get("direction")
    if direction == "unknown" and status != "material_decline":
        events.append(_event(
            None, "Revenue direction could not be established",
            "History is not longer than the recent comparison window, so there is no "
            "baseline period to compare against.",
            "revenue", CONCERN,
        ))
    return events


def _dip_and_season_events(pack: dict) -> list[dict]:
    events = []
    for episode in pack.get("dip_episodes") or []:
        span = (
            f"{episode['start_month']}"
            if episode["start_month"] == episode["end_month"]
            else f"{episode['start_month']} - {episode['end_month']}"
        )
        if episode.get("is_ongoing"):
            events.append(_event(
                episode["start_month"], f"Revenue dip began, still ongoing ({span})",
                f"{episode['months']} months below normal; trough "
                f"{_rupees(episode['trough_revenue'])} against a typical month of "
                f"{_rupees(episode['typical_monthly_revenue'])}. This dip has not recovered.",
                "revenue", CONCERN,
            ))
        else:
            events.append(_event(
                episode["start_month"], f"Revenue dip, since recovered ({span})",
                f"{episode['months']} months below normal; trough "
                f"{_rupees(episode['trough_revenue'])}. Recovered from "
                f"{episode.get('recovered_from_month')}. A resolved incident, not a live problem.",
                "revenue", REASSURING, notable=True,
            ))

    seasonality = pack.get("seasonality") or {}
    status = seasonality.get("status")
    dip_months = seasonality.get("dip_months") or []
    echo_months = seasonality.get("echo_months") or []
    if status == "confirmed":
        events.append(_event(
            echo_months[0] if echo_months else None,
            "The recent dip repeats a prior-year pattern",
            f"Dipped in {', '.join(dip_months)}; the same calendar months a year earlier "
            f"({', '.join(echo_months)}) were also below normal. Strong evidence the dip is "
            "seasonal rather than structural.",
            "revenue", REASSURING, notable=True,
        ))
    elif status == "not_present" and dip_months:
        events.append(_event(
            dip_months[0], "The recent dip has no prior-year echo",
            f"Dipped in {', '.join(dip_months)}, but the same months a year earlier were "
            "normal. Seasonality does not explain this.",
            "revenue", CONCERN,
        ))
    elif status == "no_recent_dip":
        events.append(_event(
            None, "Seasonality checked: no recent dip to explain",
            "No month in the last year fell far enough below normal to need explaining.",
            "revenue", CHECKED,
        ))
    elif status == "insufficient_history":
        events.append(_event(
            None, "Seasonality could not be checked",
            "Under 18 months of history, so there is no prior-year window to compare "
            "a recent dip against. A seasonal explanation can be neither confirmed nor ruled out.",
            "revenue", CONCERN,
        ))
    return events


def _margin_events(pack: dict) -> list[dict]:
    margin = pack.get("margin")
    if not margin:
        return []
    status = margin.get("status")
    change_pp = margin.get("margin_pct_change_pp")
    baseline_pct = margin.get("baseline_margin_pct")
    threshold = margin.get("threshold_pp")
    window_start, window_end = _recent_window(pack)
    window_label = f"{window_start} - {window_end}" if window_start else "recent window"

    if status == "insufficient_history":
        return [_event(None, "Margin could not be compared",
                       "No baseline window exists to compare the recent margin rate against.",
                       "margin", CONCERN)]

    series = ((pack.get("margin_profile") or {}).get("monthly_margin_pct_series")) or {}
    crossed = None
    if baseline_pct is not None and threshold is not None:
        floor = baseline_pct - (threshold / 100)
        crossed = _first_sustained(series, lambda v: v < floor)

    summary = (
        f"Margin rate {_pct(baseline_pct)} -> {_pct(margin.get('recent_margin_pct'))} "
        f"({_pp(change_pp)})."
    )
    how = f"Erosion threshold: {threshold}pp between the baseline and recent windows."

    if status == "erosion_detected":
        detail = summary + f" {margin.get('months_below_baseline_margin')} months sat below the normal range."
        if crossed:
            detail += f" It first fell below in {crossed} and has stayed there since."
        return [_event(crossed or window_start, "Margin rate erodes", detail,
                       "margin", CONCERN, when=crossed or window_label, method=how)]
    if status == "improvement_detected":
        return [_event(window_start, "Margin rate improves",
                       summary + " Rising margin is the opposite of a leak.",
                       "margin", REASSURING, when=window_label, method=how, notable=True)]
    return [_event(window_start, "Margin rate is stable", summary,
                   "margin", REASSURING, when=window_label, method=how)]


def _discount_events(pack: dict) -> list[dict]:
    discount = pack.get("discount")
    if not discount:
        return []
    status = discount.get("status")
    if status == "insufficient_history":
        return [_event(None, "Discount could not be compared",
                       "No baseline window exists to compare the recent discount level against.",
                       "discount", CONCERN)]

    baseline = discount.get("baseline_avg_discount_pct")
    threshold = discount.get("threshold_pp")
    window_start, window_end = _recent_window(pack)
    window_label = f"{window_start} - {window_end}" if window_start else "recent window"

    series = ((pack.get("_presentation") or {}).get("discount_profile") or {}).get(
        "monthly_avg_discount_series") or {}
    crossed = None
    if baseline is not None and threshold is not None:
        ceiling = baseline + (threshold / 100)
        crossed = _first_sustained(series, lambda v: v > ceiling)

    summary = (
        f"Average discount {_pct(baseline)} -> {_pct(discount.get('recent_avg_discount_pct'))} "
        f"({_pp(discount.get('discount_pct_change_pp'))})."
    )
    how = (f"Creep threshold: {threshold}pp. Discount is revenue-weighted, so a deep discount on "
           "a large line counts for more than the same discount on a small one.")

    if status == "creep_detected":
        detail = summary + " The same goods are being sold at a steadily deeper discount."
        if crossed:
            detail += f" It first rose above its normal range in {crossed} and has stayed there since."
        return [_event(crossed or window_start, "Discount creeps upward", detail,
                       "discount", CONCERN, when=crossed or window_label, method=how)]
    if status == "discipline_improved":
        return [_event(window_start, "Discounting tightened", summary,
                       "discount", REASSURING, when=window_label, method=how, notable=True)]
    return [_event(window_start, "Discounting is stable", summary,
                   "discount", REASSURING, when=window_label, method=how)]


def _tier_events(pack: dict) -> list[dict]:
    tier = pack.get("tier_mix")
    if not tier:
        return []
    status = tier.get("status")
    if status == "insufficient_history":
        return [_event(None, "Tier mix could not be compared",
                       "No baseline window exists to compare the recent tier mix against.",
                       "tier_mix", CONCERN)]

    baseline_shares = tier.get("baseline_share_by_tier") or {}
    recent_shares = tier.get("recent_share_by_tier") or {}
    change_pp = tier.get("high_tier_share_change_pp")
    threshold = tier.get("threshold_pp")
    window_start, window_end = _recent_window(pack)
    window_label = f"{window_start} - {window_end}" if window_start else "recent window"

    series = ((pack.get("_presentation") or {}).get("tier_mix_profile") or {}).get(
        "monthly_high_tier_share_series") or {}
    crossed = None
    baseline_high = baseline_shares.get("High")
    if baseline_high is not None and threshold is not None:
        floor = baseline_high - (threshold / 100)
        crossed = _first_sustained(series, lambda v: v < floor)

    summary = (
        f"High-tier share of revenue {_pct(baseline_high)} -> {_pct(recent_shares.get('High'))} "
        f"({_pp(change_pp)}). "
        f"Low tier {_pct(baseline_shares.get('Low'))} -> {_pct(recent_shares.get('Low'))}."
    )
    how = (f"Shift threshold: {threshold}pp. Direction decides the verdict — the same size of "
           "move upward is premiumisation, not a downgrade.")

    if status == "downgrade_detected":
        detail = summary + " Value is moving out of the high tier and being backfilled by cheaper lines."
        if crossed:
            detail += f" High-tier share first fell below its normal range in {crossed} and has stayed there since."
        return [_event(crossed or window_start, "Product mix downgrades", detail,
                       "tier_mix", CONCERN, when=crossed or window_label, method=how)]
    if status == "premiumisation_detected":
        return [_event(window_start, "Product mix moves upmarket",
                       summary + " Value is moving INTO the high tier — this is premiumisation, "
                       "the opposite of a downgrade, and is good news.",
                       "tier_mix", REASSURING, when=window_label, method=how, notable=True)]
    return [_event(window_start, "Product mix is stable", summary,
                   "tier_mix", REASSURING, when=window_label, method=how)]


def _defection_month(pack: dict, category: str, zeros: int) -> str | None:
    """The last month this category was actually bought."""
    series = ((pack.get("_presentation") or {}).get("category_monthly_revenue") or {}).get(category)
    if not series or not zeros:
        return None
    months = sorted(series)
    # The trailing `zeros` months are empty, so the last month actually
    # purchased is the one immediately before that run — not the first zero.
    index = len(months) - zeros - 1
    return months[index] if 0 <= index < len(months) else None


def _category_events(pack: dict) -> list[dict]:
    events = []
    baseline_share = (pack.get("category_mix") or {}).get("baseline_share") or {}

    for change in pack.get("category_changes") or []:
        category = change["category"]
        share = baseline_share.get(category)
        share_text = f"{_pct(share)} of this account's baseline revenue" if share is not None else "share unknown"

        if change.get("defected"):
            zeros = change.get("consecutive_months_at_zero", 0)
            last_month = _defection_month(pack, category, zeros)
            events.append(_event(
                last_month,
                f"{category} stops completely",
                f"Last purchased in {last_month or 'an earlier month'}; {zeros} consecutive months "
                f"at zero since. This category was {share_text}. A clean stop names a specific "
                "line lost, which is more actionable than a general decline.",
                "category_mix", CONCERN,
            ))
            continue

        if not change.get("change_point_detected"):
            continue

        echo = (change.get("prior_year_echo") or {}).get("status")
        precedent = change.get("seasonal_precedent")
        recovered = change.get("recovered")
        detail = (
            f"Monthly revenue {_rupees(change.get('before_monthly_median'))} -> "
            f"{_rupees(change.get('after_monthly_median'))} ({_pct(change.get('pct_decline'))}); "
            f"{share_text}. Sustained {change.get('sustained_months_since_change_point')} months."
        )
        if echo == "confirmed":
            detail += " The same dip occurred a year earlier — seasonal precedent found."
        elif echo == "not_present":
            detail += " No prior-year echo, so seasonality does not explain it."
        if recovered:
            detail += " This category has since recovered."

        how = None
        if precedent == "insufficient_history":
            how = ("The stricter multi-year seasonal check needs about three years of history and "
                   "could not run; the prior-year echo above is the check 24 months can support.")

        reads_as = REASSURING if (recovered or echo == "confirmed") else CONCERN
        if share is not None and share < 0.05 and reads_as == CONCERN:
            detail += (" This is a low-share category, where noise crosses a threshold most easily — "
                       "weak evidence on its own.")
        events.append(_event(
            change.get("change_point_month"),
            f"{category} revenue steps down {_pct(change.get('pct_decline'))}",
            detail, "category_mix", reads_as, method=how,
        ))

    disappeared = [p for p in pack.get("product_changes") or [] if p["status"] == "disappeared"]
    if disappeared:
        window_start, window_end = _recent_window(pack)
        listed = "; ".join(
            f"{p['product_id']} ({p['category']}, was {_rupees(p['baseline_monthly_avg'])}/month)"
            for p in disappeared[:5]
        )
        events.append(_event(
            window_start,
            f"{len(disappeared)} product(s) stopped being bought",
            listed + ("" if len(disappeared) <= 5 else f"; and {len(disappeared) - 5} more."),
            "category_mix", CONCERN,
            when=f"{window_start} - {window_end}" if window_start else "recent window",
        ))
    return events


def _order_pattern_events(pack: dict) -> list[dict]:
    pattern = pack.get("order_pattern") or {}
    status = pattern.get("status")
    if status in (None, "insufficient_history"):
        return []
    window_start, window_end = _recent_window(pack)
    window_label = f"{window_start} - {window_end}" if window_start else "recent window"
    behavior = pack.get("order_behavior") or {}

    summary = (
        f"Orders per month {behavior.get('order_frequency_baseline_per_month')} -> "
        f"{behavior.get('order_frequency_recent_per_month')} "
        f"({_pct(pattern.get('order_frequency_pct_change'))}); "
        f"lines per order {behavior.get('basket_width_baseline')} -> "
        f"{behavior.get('basket_width_recent')} ({_pct(pattern.get('basket_width_pct_change'))}); "
        f"average order value {_pct(pattern.get('aov_pct_change'))}."
    )
    if status == "fragmentation_detected":
        return [_event(window_start, "Ordering fragments: more orders, each smaller",
                       summary + " This is the shape of an account that has started buying part "
                       "of its basket somewhere else, and it can appear well before revenue moves.",
                       "order_pattern", CONCERN, when=window_label)]
    if status == "baskets_shrinking":
        return [_event(window_start, "Baskets are shrinking",
                       summary + " Order frequency has not risen to match, so this is not yet the "
                       "classic multi-sourcing signature.",
                       "order_pattern", CONCERN, when=window_label)]
    return [_event(window_start, "Ordering behaviour is unchanged", summary,
                   "order_pattern", REASSURING, when=window_label)]


def _data_quality_events(pack: dict) -> list[dict]:
    events = []
    quality = pack.get("data_quality") or {}

    gaps = (quality.get("gaps") or {}).get("months_with_no_orders") or []
    if gaps:
        events.append(_event(
            gaps[0], f"{len(gaps)} month(s) with no orders at all",
            f"No orders in {', '.join(gaps)}. A month with no orders is a gap in ordering, not a "
            "month of zero trading — it drags any trailing average down without any change in "
            "customer behaviour, and is not counted as a decline.",
            "data_quality", CHECKED,
        ))

    for outlier in quality.get("outlier_months") or []:
        events.append(_event(
            outlier["month"], f"Bulk month: {outlier['multiple_of_median_month']}x a typical month",
            f"{_rupees(outlier['revenue'])} in {outlier['month']}. A one-off stock-up inflates "
            "whatever window it lands in; the ordinary months after it are a return to normal, "
            "not a decline. Capped before trend fitting.",
            "data_quality", CHECKED,
        ))

    returns = quality.get("returns") or {}
    if returns.get("return_lines"):
        months = returns.get("months") or []
        events.append(_event(
            months[0] if months else None,
            f"{returns['return_lines']} return/credit line(s)",
            f"Net {_rupees(returns.get('net_return_value'))} "
            f"({_pct(returns.get('share_of_gross_revenue'))} of gross revenue) in "
            f"{', '.join(months) if months else 'the history'}. Already netted into every revenue "
            "and margin figure in this report; a credit note is not leakage.",
            "data_quality", CHECKED,
        ))
    return events


# `analysis_dimensions["returns"]` is true when the file CONTAINS return
# lines, not when returns could be analysed — it is a presence flag, unlike
# every other entry. Reporting a clean account as "returns could not be
# analysed" is both wrong and alarming, so it is excluded from coverage
# warnings; the returns handler itself is always available.
PRESENCE_ONLY_DIMENSIONS = {"returns"}


def _coverage_events(pack: dict) -> list[dict]:
    """Dimensions this input file could not support at all.

    Emitted as concerns, not omissions: a dimension that was never measured
    is the one place a confident all-clear could be wrong, and the report
    must never let "we could not look" read as "we looked and found nothing".
    """
    dimensions = pack.get("analysis_dimensions") or {}
    unavailable = [
        name for name, available in dimensions.items()
        if not available and name not in PRESENCE_ONLY_DIMENSIONS
    ]
    if not unavailable:
        return []
    return [_event(
        None, f"{len(unavailable)} dimension(s) could not be analysed",
        "This file carries no data for: "
        + ", ".join(n.replace("_", " ") for n in unavailable)
        + ". These were not measured — that is not the same as finding them unchanged.",
        "coverage", CONCERN,
    )]


NO_BASELINE_HEADLINES = {
    "Margin could not be compared": "margin",
    "Discount could not be compared": "discount",
    "Tier mix could not be compared": "tier mix",
}


def _merge_no_baseline_events(events: list[dict]) -> list[dict]:
    """Collapse the per-dimension "could not be compared" rows into one.

    On an account too new to have a baseline window, margin, discount and
    tier mix each report the same single cause. Printed separately they read
    as three findings and crowd out the one thing that matters — that the
    history is too short to judge anything. Merged, the point is made once.
    """
    absent = [e for e in events if e["headline"] in NO_BASELINE_HEADLINES]
    if len(absent) < 2:
        return events

    names = [NO_BASELINE_HEADLINES[e["headline"]] for e in absent]
    listed = ", ".join(names[:-1]) + f" and {names[-1]}"
    merged = _event(
        None,
        f"{listed.capitalize()} could not be compared",
        "No baseline window exists on this account, so none of these dimensions can be "
        "measured against a prior period. This is a limit of the history available, not a "
        "finding that they are unchanged.",
        "coverage", CONCERN,
    )
    first = events.index(absent[0])
    remaining = [e for e in events if e["headline"] not in NO_BASELINE_HEADLINES]
    return remaining[:first] + [merged] + remaining[first:]


def build_timeline(evidence_pack: dict) -> list[dict]:
    """The full dated evidence log, in chronological order.

    Undated events (checks that span the whole history, or that could not be
    anchored to a month) sort to the end rather than being dropped — an
    unanswerable question is part of the case too.
    """
    events = (
        _history_events(evidence_pack)
        + _revenue_events(evidence_pack)
        + _dip_and_season_events(evidence_pack)
        + _margin_events(evidence_pack)
        + _discount_events(evidence_pack)
        + _tier_events(evidence_pack)
        + _category_events(evidence_pack)
        + _order_pattern_events(evidence_pack)
        + _data_quality_events(evidence_pack)
        + _coverage_events(evidence_pack)
    )
    # "~" sorts after any "YYYY-MM", so undated events land at the end
    # without needing a separate list or a None-safe comparator.
    return _merge_no_baseline_events(sorted(events, key=lambda e: e["month"] or "~"))


def timeline_counts(timeline: list[dict]) -> dict:
    """How many events of each kind — the one-line summary of the case."""
    counts = {kind: 0 for kind in READS_AS_ORDER}
    for event in timeline:
        if event["reads_as"] in counts:
            counts[event["reads_as"]] += 1
    return counts
