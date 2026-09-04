"""
PDF export for a finished report.

The job of this document is not to restate the verdict — the UI already
does that. It is to make the verdict *arguable*: every claim carries the
date it was observed, the figure it rests on, and the threshold it was
judged against, so a reader who disagrees can point at the row they
disagree with. That is also the difference between a report a revenue team
can act on and a classification they have to take on faith.

Two depths, chosen by the caller:

  brief  ~2 pages. The verdict, the money, the dated timeline, the actions.
  full   Everything, plus per-dimension evidence, the explanations that
         were ruled out, product-level attribution, and a provenance
         appendix carrying every monthly series.

Both are built from the same report dict. Nothing here computes an
analytical figure; every number printed was produced by a pipeline stage
and is reachable from `report["evidence"]`.

Currency is written "Rs" throughout: reportlab's built-in Helvetica has no
glyph for U+20B9 and would silently print a black box.
"""

from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pipeline.timeline import PRESENCE_ONLY_DIMENSIONS

# This report is tables only, by design. Charts were built and reviewed, then
# dropped: every series they showed is available here as figures a reader can
# quote in an email, and a table needs no rendering dependency on the demo
# machine. The monthly appendix carries the series the line charts plotted.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
NEUTRAL_FILL = "#f0efec"

INK = colors.HexColor(INK_PRIMARY)
INK_SOFT = colors.HexColor(INK_SECONDARY)
MUTED = colors.HexColor(INK_MUTED)
RULE = colors.HexColor("#e1e0d9")
PANEL = colors.HexColor(NEUTRAL_FILL)

STATUS_COLORS = {
    "concern": colors.HexColor("#d03b3b"),
    "reassuring": colors.HexColor("#0ca30c"),
    "checked": colors.HexColor(INK_SECONDARY),
    "context": colors.HexColor(INK_MUTED),
}

VERDICT_COLORS = {
    "leakage_detected": colors.HexColor("#d03b3b"),
    "healthy": colors.HexColor("#0ca30c"),
    "insufficient_data": colors.HexColor("#b07800"),
}

READS_AS_LABELS = {
    "concern": "Concern",
    "reassuring": "Reassuring",
    "checked": "Checked",
    "context": "Context",
}

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 16 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

BRIEF = "brief"
FULL = "full"


# --- styles ----------------------------------------------------------------


def _styles() -> dict:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9,
        leading=12.5, textColor=INK, alignment=TA_LEFT, spaceAfter=0,
    )
    return {
        "title": ParagraphStyle("Title2", parent=body, fontName="Helvetica-Bold",
                                fontSize=17, leading=20, spaceAfter=2),
        "subtitle": ParagraphStyle("Subtitle", parent=body, fontSize=9.5,
                                   textColor=INK_SOFT, spaceAfter=10),
        "h1": ParagraphStyle("H1", parent=body, fontName="Helvetica-Bold",
                             fontSize=11.5, leading=14, spaceBefore=13, spaceAfter=5),
        "h2": ParagraphStyle("H2", parent=body, fontName="Helvetica-Bold",
                             fontSize=9.5, leading=12, spaceBefore=8, spaceAfter=3),
        "body": body,
        "lead": ParagraphStyle("Lead", parent=body, fontSize=10, leading=14.5),
        "small": ParagraphStyle("Small", parent=body, fontSize=7.6, leading=10,
                                textColor=INK_SOFT),
        "muted": ParagraphStyle("Muted", parent=body, fontSize=7.6, leading=10,
                                textColor=MUTED),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=8, leading=10.5),
        "cell_small": ParagraphStyle("CellSmall", parent=body, fontSize=7.4,
                                     leading=9.5, textColor=INK_SOFT),
        "bullet": ParagraphStyle("Bullet", parent=body, fontSize=9, leading=12.5,
                                 leftIndent=10, bulletIndent=1, spaceAfter=2.5),
    }


# --- formatting helpers ----------------------------------------------------


# The built-in Helvetica can only encode WinAnsi (cp1252). Anything outside
# it renders as a black notdef box — reportlab does not warn. Model output
# routinely contains characters that are NOT in cp1252 even though they look
# ordinary: the non-breaking hyphen below is what turned "Re-run" into
# "Re<box>run" in a live report. Map the ones with obvious ASCII equivalents
# rather than dropping them, so "->" survives as an arrow and not as a hole.
UNICODE_FALLBACKS = {
    "‐": "-", "‑": "-", "‒": "-", "―": "-",  # hyphens/dashes
    "‘": "'", "’": "'", "‚": "'", "‛": "'",  # single quotes
    "“": '"', "”": '"', "„": '"',                 # double quotes
    "•": "-", "‣": "-", "▪": "-",                 # bullets
    "→": "->", "←": "<-", "↔": "<->",             # arrows
    "≤": "<=", "≥": ">=", "≠": "!=",              # comparisons
    "₹": "Rs", " ": " ", "​": "",                 # rupee, spaces
    "′": "'", "″": '"', "⁄": "/",
}


def _ascii_safe(text: str) -> str:
    """Make text renderable by a standard PDF font.

    Known characters are mapped to their ASCII equivalent; anything still
    outside cp1252 is dropped rather than printed as a box, because a
    missing character reads as a typo and a black box reads as a broken
    document.
    """
    for source, replacement in UNICODE_FALLBACKS.items():
        if source in text:
            text = text.replace(source, replacement)
    if text.isascii():
        return text
    return text.encode("cp1252", errors="ignore").decode("cp1252")


def _text(value) -> str:
    """Escape and sanitize anything that reaches a Paragraph.

    Narrative text, cited facts and category names all originate outside
    this module — a stray '&' or '<' in any of them would otherwise abort
    the export with a parse error at the worst possible moment, and a stray
    Unicode dash would silently print as a black box.
    """
    return escape(_ascii_safe("" if value is None else str(value)))


def _rupees(value) -> str:
    if value is None:
        return "n/a"
    return f"Rs {value:,.0f}"


def _pct(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _pp(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}pp"


def _humanize(value) -> str:
    return str(value or "").replace("_", " ").strip().title()


def _hex(color) -> str:
    """'#rrggbb' for a reportlab colour.

    `Color.hexval()` returns '0xrrggbb', which the inline-markup parser
    rejects — it wants CSS-style hex.
    """
    return "#" + color.hexval()[2:]


# --- reusable flowables ----------------------------------------------------


def _rule(space_before: float = 2, space_after: float = 6) -> Table:
    table = Table([[""]], colWidths=[CONTENT_WIDTH], rowHeights=[0.6])
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.6, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), space_before),
        ("BOTTOMPADDING", (0, 0), (-1, -1), space_after),
    ]))
    return table


def _key_value_panel(rows: list[tuple[str, str]], styles: dict, columns: int = 4) -> Table:
    """A KPI strip: small muted label above a bold value, repeated across.

    Used instead of a sentence for the four numbers a reader looks for
    first, so they can be found without reading anything.
    """
    cells = []
    for label, value in rows:
        cells.append([
            Paragraph(f'<font size="6.8" color="{INK_MUTED}">{_text(label.upper())}</font>',
                      styles["small"]),
            Paragraph(f'<b>{_text(value)}</b>', styles["cell"]),
        ])
    while len(cells) % columns:
        cells.append([Paragraph("", styles["small"]), Paragraph("", styles["cell"])])

    grid = []
    for start in range(0, len(cells), columns):
        chunk = cells[start:start + columns]
        grid.append([c[0] for c in chunk])
        grid.append([c[1] for c in chunk])

    width = CONTENT_WIDTH / columns
    table = Table(grid, colWidths=[width] * columns)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
    ]
    for row in range(0, len(grid), 2):
        style += [("TOPPADDING", (0, row), (-1, row), 7),
                  ("BOTTOMPADDING", (0, row), (-1, row), 1),
                  ("TOPPADDING", (0, row + 1), (-1, row + 1), 0),
                  ("BOTTOMPADDING", (0, row + 1), (-1, row + 1), 7)]
    table.setStyle(TableStyle(style))
    return table


def _data_table(header: list[str], rows: list[list], widths: list[float]) -> Table:
    table = Table([header] + rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7.4),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK_SOFT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, colors.HexColor("#f0efec")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]))
    return table


def _headed(heading, body: list) -> list:
    """Bind a section heading to its first line of content.

    Without this, a heading that lands near a page foot is printed alone and
    its section starts overleaf — which reads as an empty section.
    """
    if not body:
        return [heading]
    return [KeepTogether([heading, body[0]])] + body[1:]


def _bullets(items, styles: dict, style_key: str = "bullet") -> list:
    return [
        Paragraph(_text(item), styles[style_key], bulletText="•")
        for item in items or []
    ]


# --- sections --------------------------------------------------------------


def _header(report: dict, styles: dict, depth: str) -> list:
    evidence = report.get("evidence") or {}
    history = evidence.get("history") or {}
    account = report.get("account_id", "unknown")
    name = report.get("account_name")
    heading = f"{account} - {name}" if name else str(account)

    identity_bits = []
    if evidence.get("region"):
        identity_bits.append(f"Region: {evidence['region']}")
    manager = evidence.get("account_manager")
    if manager:
        identity_bits.append(
            "Account manager: " + (", ".join(manager) if isinstance(manager, list) else str(manager))
        )
    if history.get("start_date"):
        identity_bits.append(f"History: {history['start_date']} to {history.get('end_date')}")
    identity_bits.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    flow = [
        Paragraph("Revenue Leakage Investigation", styles["subtitle"]),
        Paragraph(_text(heading), styles["title"]),
        Paragraph(_text(" &nbsp;|&nbsp; ".join(identity_bits)).replace(
            "&amp;nbsp;", "&nbsp;"), styles["small"]),
        Spacer(1, 9),
    ]
    if report.get("_cache_metadata"):
        flow.append(Paragraph(
            "<b>Cached demo report.</b> The deterministic analysis is real, but the Stage 4 "
            "judgement in this document is a scripted stand-in, not a live model response.",
            styles["small"]))
        flow.append(Spacer(1, 6))
    return flow


def _verdict_block(report: dict, styles: dict) -> list:
    verdict = report.get("verdict", "unknown")
    deferring = report.get("defer")
    priority = report.get("prioritization") or {}
    impact = report.get("financial_impact") or {}

    if deferring:
        headline, color = "Deferred to a human", VERDICT_COLORS["insufficient_data"]
    else:
        headline = _humanize(verdict)
        color = VERDICT_COLORS.get(verdict, MUTED)

    banner = Table([[Paragraph(
        f'<font color="{_hex(color)}" size="13"><b>{_text(headline)}</b></font>',
        styles["body"])]], colWidths=[CONTENT_WIDTH])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LINEBEFORE", (0, 0), (0, -1), 3, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    severity = impact.get("overall_severity_pct_of_baseline")
    if deferring:
        # "0.0%" would read as "we measured the loss and it was nothing",
        # which is the opposite of what deferring means.
        severity_text = "Not assessed"
    elif severity:
        severity_text = _pct(severity)
    else:
        severity_text = "None sized"

    metrics = [
        ("Confidence", _humanize(report.get("confidence"))),
        ("Nature", _humanize(report.get("temporary_or_structural"))),
        ("Priority", str(priority.get("priority", "n/a"))),
        ("Severity", severity_text),
    ]

    flow = [banner, Spacer(1, 7), _key_value_panel(metrics, styles), Spacer(1, 10)]

    if report.get("leak_dimensions"):
        flow.append(Paragraph(
            "<b>Leaking through:</b> "
            + _text(", ".join(_humanize(d) for d in report["leak_dimensions"])),
            styles["body"]))
        flow.append(Spacer(1, 5))

    flow.append(Paragraph(_text(report.get("narrative")), styles["lead"]))
    return flow


def _impact_section(report: dict, styles: dict, depth: str = FULL) -> list:
    impact = report.get("financial_impact") or {}
    priority = report.get("prioritization") or {}
    margin = impact.get("margin_impact")
    account_level = impact.get("account_level_revenue_impact")
    per_category = [c for c in impact.get("per_category") or [] if c.get("quantifiable")]

    if not (margin or account_level or per_category):
        # A leakage verdict with nothing sized is a real state, not an
        # omission — say so, rather than leaving a blank where money should
        # be and letting the reader assume the loss is zero.
        if report.get("verdict") == "leakage_detected" and not report.get("defer"):
            return [
                Paragraph("Financial impact", styles["h1"]),
                Paragraph(
                    "The leak was identified but could not be sized in rupees: it is a change in "
                    "behaviour (mix, discounting or order pattern) with no quantifiable revenue "
                    "step-down attached to a named category. The dimension rows below carry the "
                    "measured movement.", styles["body"]),
            ]
        return []

    flow = [Paragraph("Financial impact", styles["h1"])]

    tiles = []
    revenue_at_risk = impact.get("total_monthly_revenue_at_risk") or 0
    if revenue_at_risk:
        tiles.append(("Monthly revenue at risk", _rupees(revenue_at_risk)))
    if margin:
        tiles.append(("Monthly margin at risk", _rupees(margin["monthly_margin_at_risk"])))
    projection = priority.get("churn_risk_projection") or {}
    if projection.get("projected_12_month_loss_if_unaddressed"):
        tiles.append(("12-month revenue exposure",
                      _rupees(projection["projected_12_month_loss_if_unaddressed"])))
    if projection.get("projected_12_month_margin_loss_if_unaddressed"):
        tiles.append(("12-month margin exposure",
                      _rupees(projection["projected_12_month_margin_loss_if_unaddressed"])))
    if tiles:
        flow += [_key_value_panel(tiles, styles, columns=min(len(tiles), 4)), Spacer(1, 5)]
        # Only worth saying when both figures are actually on the page; on a
        # margin-only leak it warns against an addition nobody could make.
        if revenue_at_risk and margin:
            flow.append(Paragraph(
                "Revenue at risk and margin at risk overlap - margin is a slice of revenue - and "
                "are never added together. Severity is the worse of the two ratios.",
                styles["muted"]))

    if per_category:
        flow += [Spacer(1, 8), Paragraph("By category", styles["h2"])]
        rows = [[
            Paragraph(_text(c["category"]), styles["cell"]),
            Paragraph(_rupees(c["monthly_revenue_at_risk"]), styles["cell"]),
            Paragraph(_rupees(c["annualized_run_rate_loss"]), styles["cell"]),
            Paragraph(str(c.get("persistence_months", 0)), styles["cell"]),
            Paragraph(_pct(c.get("category_baseline_revenue_share")), styles["cell"]),
        ] for c in per_category]
        flow.append(_data_table(
            ["Category", "Per month", "Annualised", "Months", "Baseline share"],
            rows,
            [CONTENT_WIDTH * w for w in (0.34, 0.17, 0.19, 0.13, 0.17)],
        ))

    for c in impact.get("per_category") or []:
        if not c.get("quantifiable"):
            flow.append(Paragraph(
                f"{_text(c['category'])}: attributed but not quantifiable - {_text(c.get('reason'))}",
                styles["muted"]))

    if account_level:
        flow += [Spacer(1, 7), Paragraph(
            f"<b>Whole-account decline:</b> {_rupees(account_level['baseline_monthly_revenue'])}"
            f"/month to {_rupees(account_level['recent_monthly_revenue'])}/month.",
            styles["body"])]
        if depth == FULL:
            flow.append(Paragraph(_text(account_level.get("basis")), styles["muted"]))

    if margin:
        flow += [Spacer(1, 7), Paragraph(
            f"<b>Margin rate:</b> {_pct(margin['baseline_margin_pct'])} to "
            f"{_pct(margin['recent_margin_pct'])} ({_pp(margin.get('margin_pct_erosion_pp'))}).",
            styles["body"])]
        if depth == FULL:
            flow.append(Paragraph(_text(margin.get("basis")), styles["muted"]))

    if priority.get("reason"):
        flow += [Spacer(1, 7),
                 Paragraph(f"<b>Priority {priority.get('priority')}</b> - "
                           f"{_text(priority['reason'])}", styles["body"])]
    return flow


def _timeline_section(report: dict, styles: dict, depth: str = FULL) -> list:
    """The dated evidence log.

    In the brief this is the findings only, with everything that was checked
    and found normal compressed into a single line underneath. A commercial
    reader needs "what is wrong, since when, and how much"; four rows of
    threshold arithmetic confirming that nothing else moved buries that.
    The full dossier keeps every row and every threshold.
    """
    timeline = report.get("evidence_timeline") or []
    if not timeline:
        return []

    if depth == BRIEF:
        # Anything that MOVED, in either direction — an account trading up is
        # a finding too. Only the rows reporting that nothing changed get
        # folded into the summary line.
        shown = [e for e in timeline if e.get("notable")]
        cleared = [e for e in timeline
                   if not e.get("notable") and e["reads_as"] in ("reassuring", "checked")]
    else:
        shown, cleared = timeline, []

    rows = []
    for event in shown:
        color = STATUS_COLORS.get(event["reads_as"], MUTED)
        detail = _text(event["detail"])
        if depth == FULL and event.get("method"):
            detail += (f'<br/><font color="{INK_MUTED}">How this was judged: '
                       f'{_text(event["method"])}</font>')
        rows.append([
            Paragraph(f'<font size="7.4">{_text(event["when"])}</font>', styles["cell_small"]),
            Paragraph(
                f'<b>{_text(event["headline"])}</b><br/>'
                f'<font size="7.4" color="{INK_SECONDARY}">{detail}</font>',
                styles["cell"]),
            Paragraph(
                f'<font color="{_hex(color)}" size="7.4"><b>'
                f'{READS_AS_LABELS.get(event["reads_as"], "")}</b></font>',
                styles["cell_small"]),
        ])

    if depth == BRIEF:
        intro = ("What moved on this account, dated to the month it started. Everything else was "
                 "checked and found normal.")
    else:
        intro = ("Every row is a fact computed by the deterministic stages, dated to the month it "
                 "was observed and marked for how it bears on the verdict. &#8220;Checked&#8221; "
                 "rows are innocent explanations that were tested and did not account for what "
                 "was found.")

    flow = [
        Paragraph("What changed, and when", styles["h1"]),
        Paragraph(intro, styles["small"]),
        Spacer(1, 7),
    ]
    if rows:
        flow.append(_data_table(
            ["When", "What happened", "Reads as"],
            rows,
            [CONTENT_WIDTH * w for w in (0.16, 0.70, 0.14)],
        ))
    else:
        flow.append(Paragraph("Nothing on this account moved outside its normal range.",
                              styles["body"]))

    if cleared:
        # Named, not just counted: "we checked seasonality" is the answer to
        # the first question a sceptical reader asks, and it costs one line.
        flow += [
            Spacer(1, 7),
            Paragraph(
                "<b>Also checked and found normal:</b> "
                + _text("; ".join(e["headline"] for e in cleared))
                + ". Full detail and the thresholds behind every call are in the full report.",
                styles["small"]),
        ]
    return flow


def _dimension_rows(report: dict) -> list[tuple]:
    """One row per detection dimension: what it was, what it is, the
    threshold, and the call. The threshold column is the point — it turns
    'margin fell' into a claim the reader can check."""
    evidence = report.get("evidence") or {}
    dimensions = report.get("analysis_dimensions") or {}
    rows = []

    decline = evidence.get("revenue_decline") or {}
    overall = evidence.get("overall_revenue") or {}
    rows.append((
        "Revenue", _humanize(decline.get("status")),
        f"{_rupees(overall.get('baseline_monthly_median'))}/mo",
        f"{_rupees(overall.get('recent_monthly_rate'))}/mo",
        f"half-over-half {_pct(decline.get('first_half_vs_second_half_pct_change'))}",
        "material at -15% and -1.5%/mo",
    ))

    margin = evidence.get("margin")
    if margin and dimensions.get("margin"):
        rows.append((
            "Margin", _humanize(margin.get("status")),
            _pct(margin.get("baseline_margin_pct")), _pct(margin.get("recent_margin_pct")),
            _pp(margin.get("margin_pct_change_pp")), f"{margin.get('threshold_pp')}pp",
        ))
    else:
        rows.append(("Margin", "Not measured", "-", "-", "-", "no margin data in this file"))

    discount = evidence.get("discount")
    if discount and dimensions.get("discount"):
        rows.append((
            "Discount", _humanize(discount.get("status")),
            _pct(discount.get("baseline_avg_discount_pct")),
            _pct(discount.get("recent_avg_discount_pct")),
            _pp(discount.get("discount_pct_change_pp")), f"{discount.get('threshold_pp')}pp",
        ))
    else:
        rows.append(("Discount", "Not measured", "-", "-", "-", "no discount data in this file"))

    tier = evidence.get("tier_mix")
    if tier and dimensions.get("tier_mix"):
        baseline_shares = tier.get("baseline_share_by_tier") or {}
        recent_shares = tier.get("recent_share_by_tier") or {}
        rows.append((
            "Tier mix", _humanize(tier.get("status")),
            f"High {_pct(baseline_shares.get('High'))}",
            f"High {_pct(recent_shares.get('High'))}",
            _pp(tier.get("high_tier_share_change_pp")), f"{tier.get('threshold_pp')}pp",
        ))
    else:
        rows.append(("Tier mix", "Not measured", "-", "-", "-", "no product tier in this file"))

    pattern = evidence.get("order_pattern") or {}
    behavior = evidence.get("order_behavior") or {}
    rows.append((
        "Order pattern", _humanize(pattern.get("status")),
        f"{behavior.get('order_frequency_baseline_per_month')}/mo, "
        f"{behavior.get('basket_width_baseline')} lines",
        f"{behavior.get('order_frequency_recent_per_month')}/mo, "
        f"{behavior.get('basket_width_recent')} lines",
        f"width {_pct(pattern.get('basket_width_pct_change'))}",
        "+30% freq and -20% width",
    ))

    changes = evidence.get("category_changes") or []
    defected = [c["category"] for c in changes if c.get("defected")]
    stepped = [c for c in changes if c.get("change_point_detected")]
    rows.append((
        "Category mix",
        "Defection" if defected else ("Step-down" if stepped else "Stable"),
        f"{len(changes)} categories",
        (f"{len(defected)} stopped" if defected else f"{len(stepped)} stepped down"),
        ", ".join(defected) if defected else
        (", ".join(c["category"] for c in stepped[:2]) if stepped else "-"),
        "3 months at zero = defection",
    ))
    return rows


def _dimensions_section(report: dict, styles: dict) -> list:
    rows = [
        [Paragraph(f"<b>{_text(name)}</b>", styles["cell"]),
         Paragraph(_text(status), styles["cell"]),
         Paragraph(_text(baseline), styles["cell_small"]),
         Paragraph(_text(recent), styles["cell_small"]),
         Paragraph(_text(change), styles["cell_small"]),
         Paragraph(_text(threshold), styles["cell_small"])]
        for name, status, baseline, recent, change, threshold in _dimension_rows(report)
    ]
    return [
        Paragraph("Evidence by dimension", styles["h1"]),
        Paragraph(
            "Six dimensions are profiled in parallel. Two of them - margin and discount - can "
            "show a leak while revenue stays perfectly flat, which is why a revenue-only review "
            "is not sufficient. A dimension marked &#8220;not measured&#8221; was absent from the "
            "input file; that is not the same as finding it unchanged.", styles["small"]),
        Spacer(1, 7),
        _data_table(
            ["Dimension", "Finding", "Baseline", "Recent", "Change", "Threshold"],
            rows,
            [CONTENT_WIDTH * w for w in (0.15, 0.16, 0.18, 0.18, 0.16, 0.17)],
        ),
    ]


def _ruled_out_section(report: dict, styles: dict) -> list:
    """The alternatives that were tested. On a NO FLAG account this section
    IS the finding; on a FLAG account it is what makes the finding safe to
    act on."""
    # Only "checked" events. The "reassuring" ones are already in the
    # timeline with identical wording, and reprinting them here made this
    # section read as padding rather than as the list of alternatives that
    # were actively tested.
    checked = [e for e in report.get("evidence_timeline") or []
               if e["reads_as"] == "checked"]
    if not checked:
        return []
    rows = [[
        Paragraph(f'<b>{_text(e["headline"])}</b>', styles["cell"]),
        Paragraph(_text(e["detail"]), styles["cell_small"]),
    ] for e in checked]
    return [
        Paragraph("Alternative explanations considered", styles["h1"]),
        Paragraph(
            "A decline has many innocent causes. Each of these was tested against this account's "
            "own history before any conclusion was drawn.", styles["small"]),
        Spacer(1, 7),
        _data_table(["Checked", "What the data showed"], rows,
                    [CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.70]),
    ]


def _attribution_section(report: dict, styles: dict) -> list:
    evidence = report.get("evidence") or {}
    categories = report.get("attributed_categories") or []
    products = evidence.get("product_changes") or []

    flow = [Paragraph("Where the value went", styles["h1"])]
    if categories:
        flow.append(Paragraph(
            "<b>Attributed categories:</b> " + _text(", ".join(categories)), styles["body"]))
    elif report.get("verdict") == "leakage_detected" and not report.get("defer"):
        flow.append(Paragraph(
            "<b>No single category is responsible.</b> The loss is spread across the book, so no "
            "category is named. Inventing a scapegoat category for a diffuse decline would send "
            "the account team after the wrong thing.", styles["body"]))
    else:
        flow.append(Paragraph("No categories attributed.", styles["body"]))

    if products:
        flow += [Spacer(1, 7), Paragraph("Product-level movement", styles["h2"])]
        rows = [[
            Paragraph(_text(p["product_id"]), styles["cell"]),
            Paragraph(_text(p["category"]), styles["cell_small"]),
            Paragraph(_rupees(p["baseline_monthly_avg"]), styles["cell_small"]),
            Paragraph(_rupees(p["recent_monthly_avg"]), styles["cell_small"]),
            Paragraph(_humanize(p["status"]), styles["cell_small"]),
        ] for p in products[:12]]
        flow.append(_data_table(
            ["Product", "Category", "Baseline/mo", "Recent/mo", "Status"],
            rows,
            [CONTENT_WIDTH * w for w in (0.30, 0.24, 0.16, 0.16, 0.14)],
        ))
    return flow


def _defer_section(report: dict, styles: dict) -> list:
    if not report.get("defer") and report.get("verdict") != "insufficient_data":
        return []
    needed = report.get("data_needed_if_deferring") or []
    flow = _headed(
        Paragraph("Why this was not called", styles["h1"]),
        [Paragraph(
            "This account was deferred to a human rather than classified. Deferring is a designed "
            "outcome, not a failure: where the history cannot separate a temporary dip from a "
            "structural loss, a confident verdict would be a guess presented as a finding.",
            styles["body"])],
    )
    if needed:
        flow += [Spacer(1, 6), Paragraph("What would resolve it", styles["h2"])]
        flow += _bullets(needed, styles)
    return flow


def _actions_section(report: dict, styles: dict) -> list:
    actions = report.get("recommended_actions") or []
    if not actions:
        return []
    return _headed(Paragraph("Recommended actions", styles["h1"]),
                   _bullets(actions, styles))


def _cited_section(report: dict, styles: dict) -> list:
    facts = report.get("cited_evidence") or []
    if not facts:
        return []
    return _headed(
        Paragraph("Facts the verdict rests on", styles["h1"]),
        [
            Paragraph(
                "Cited by the investigation agent. The agent selects and weighs facts; it does not "
                "compute them - every figure it cites was produced by the deterministic stages and "
                "appears elsewhere in this document.", styles["small"]),
            Spacer(1, 5),
        ] + _bullets(facts, styles))


def _coverage_section(report: dict, styles: dict, depth: str = FULL) -> list:
    dimensions = report.get("analysis_dimensions") or {}
    unavailable = [n for n, available in dimensions.items()
                   if not available and n not in PRESENCE_ONLY_DIMENSIONS]
    sufficiency = report.get("data_sufficiency") or {}
    flags = sufficiency.get("flags") or []

    # In the brief this section earns its place only when there IS a
    # limitation. "Data sufficiency: sufficient" is a line saying nothing was
    # wrong with the data, which is not news; an unmeasured dimension or a
    # thin history very much is.
    if depth == BRIEF and not unavailable and not flags and sufficiency.get("label") == "sufficient":
        return []

    body = []
    body.append(Paragraph(
        f"Data sufficiency: <b>{_text(sufficiency.get('label'))}</b> "
        f"(score {sufficiency.get('score')}) - {sufficiency.get('history_months')} months, "
        f"{sufficiency.get('order_count')} orders, {sufficiency.get('category_count')} categories.",
        styles["body"]))
    if flags:
        body.append(Paragraph(
            "Flags: " + _text(", ".join(f.replace("_", " ") for f in flags)) + ".",
            styles["body"]))
    if unavailable:
        body.append(Spacer(1, 4))
        body.append(Paragraph(
            "<b>Not analysed</b> (absent from this file): "
            + _text(", ".join(n.replace("_", " ") for n in unavailable))
            + ". These dimensions were not measured. That is not a finding that they are healthy.",
            styles["body"]))
    return _headed(Paragraph("Coverage and limitations", styles["h1"]), body)


def _provenance_section(report: dict, styles: dict) -> list:
    evidence = report.get("evidence") or {}
    overall = evidence.get("overall_revenue") or {}
    series = overall.get("monthly_series") or {}

    flow = [
        Paragraph("Appendix: provenance", styles["h1"]),
        Paragraph(
            "Every figure in this report is produced by deterministic code. Stage 1 resolves and "
            "validates the input file; Stage 2 builds the account's baseline; Stage 3 runs the "
            "detectors and their thresholds; Stage 4 - the single model call in the pipeline - "
            "weighs those facts into a verdict but computes no numbers of its own; Stages 5 to 7 "
            "size the impact, rank it, and assemble this document. The reference answer key used "
            "to score this system in testing is never visible to the agent.", styles["small"]),
    ]

    if not series:
        return flow

    # Every monthly series the pipeline computed, side by side. This is the
    # table that replaces the line charts: the same numbers, quotable, and
    # checkable against the dated claims in the timeline.
    presentation = evidence.get("_presentation") or {}
    optional = [
        ("Margin %", (evidence.get("margin_profile") or {}).get("monthly_margin_pct_series") or {}),
        ("Discount %", (presentation.get("discount_profile") or {}).get(
            "monthly_avg_discount_series") or {}),
        ("High tier %", (presentation.get("tier_mix_profile") or {}).get(
            "monthly_high_tier_share_series") or {}),
    ]
    # A column of nothing but dashes tells the reader less than its absence
    # does — the coverage section already names what was not measured.
    optional = [(label, values) for label, values in optional
                if any(v is not None for v in values.values())]

    months = sorted(series)
    rows = [
        [Paragraph(month, styles["cell_small"]),
         Paragraph(_rupees(series[month]), styles["cell_small"])]
        + [Paragraph(_pct(values.get(month)), styles["cell_small"]) for _label, values in optional]
        for month in months
    ]

    header = ["Month", "Revenue"] + [label for label, _values in optional]
    weights = [0.16, 0.24] + [(1 - 0.40) / len(optional)] * len(optional) if optional else [0.3, 0.7]
    flow += [
        Spacer(1, 8),
        Paragraph("Monthly series", styles["h2"]),
        Paragraph(
            "The figures behind every dated claim in the timeline. Margin, discount and high-tier "
            "share are the three dimensions that can move while revenue stays flat.",
            styles["small"]),
        Spacer(1, 5),
        _data_table(header, rows, [CONTENT_WIDTH * w for w in weights]),
    ]
    return flow


# --- page furniture --------------------------------------------------------


def _page_decoration(canvas, doc, account_label: str) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 10 * mm, account_label)
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 10 * mm, f"Page {canvas.getPageNumber()}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 13 * mm, PAGE_WIDTH - MARGIN, 13 * mm)
    canvas.restoreState()


# --- entry point -----------------------------------------------------------


def build_pdf(report: dict, depth: str = FULL) -> bytes:
    """Render a finished report to PDF bytes.

    `depth` is BRIEF (verdict, money, timeline, actions) or FULL (everything,
    including per-dimension evidence, ruled-out explanations and the
    provenance appendix).
    """
    if depth not in (BRIEF, FULL):
        raise ValueError(f"Unknown report depth {depth!r}; expected {BRIEF!r} or {FULL!r}.")

    styles = _styles()
    buffer = io.BytesIO()
    account_label = report.get("account_id", "")
    if report.get("account_name"):
        account_label = f"{account_label} - {report['account_name']}"

    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=20 * mm,
        title=f"Revenue leakage investigation - {account_label}",
        author="Revenue Leakage Investigator",
        subject="Account revenue leakage analysis",
    )

    flow = []
    flow += _header(report, styles, depth)
    flow += _verdict_block(report, styles)
    flow.append(Spacer(1, 4))
    flow += _defer_section(report, styles)
    flow += _impact_section(report, styles, depth)
    flow.append(_rule())
    flow += _timeline_section(report, styles, depth)

    if depth == FULL:
        flow.append(PageBreak())
        flow += _dimensions_section(report, styles)
        flow += _ruled_out_section(report, styles)
        flow += _attribution_section(report, styles)

    if depth == FULL:
        # The agent's own citations are variable-name-level detail ("
        # margin_pct_change_pp -11.08, exceeds threshold 3pp"). They are the
        # reasoning trail a reviewer wants and noise to an account owner, who
        # has the same facts in English in the timeline above.
        flow += _cited_section(report, styles)
    flow += _actions_section(report, styles)
    flow += _coverage_section(report, styles, depth)

    if depth == FULL:
        flow += _provenance_section(report, styles)

    decorate = lambda canvas, doc: _page_decoration(canvas, doc, account_label)  # noqa: E731
    document.build(flow, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()


def pdf_filename(report: dict, depth: str = FULL) -> str:
    account = str(report.get("account_id", "account")).replace(" ", "_")
    return f"revenue_leakage_{account}_{depth}_{datetime.now():%Y%m%d}.pdf"
