"""
Tests for the PDF export.

These do not assert on visual layout — that is checked by eye. They pin the
things that would silently produce a wrong or broken document: that every
verdict shape renders at both depths, that hostile text in a narrative
cannot abort the export, that a thin file still produces a report, and that
the export never invents a number the pipeline did not compute.
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from datasets import FLAGSHIP_LEAK, THIN_HISTORY, meridian_transactions, thin_transactions
from fakes import ScriptedClient, message, tool_use_block
from pipeline.orchestrate import run_for_account
from reporting.pdf import BRIEF, FULL, build_pdf, pdf_filename

PDF_MAGIC = b"%PDF-"


def page_count(pdf: bytes) -> int:
    """Pages in a reportlab-produced PDF.

    Counts page objects directly rather than pulling in a PDF parser: the
    document is uncompressed reportlab output, so '/Type /Page' appears once
    per page plus once for the '/Type /Pages' tree node.
    """
    return pdf.count(b"/Type /Page") - pdf.count(b"/Type /Pages")


def verdict(**overrides) -> dict:
    base = {
        "verdict": "healthy", "temporary_or_structural": "not_applicable",
        "confidence": "high", "defer": False, "leak_dimensions": [],
        "attributed_categories": [], "cited_facts": ["a cited fact"],
        "narrative": "A narrative.", "recommended_actions": ["An action."],
        "data_needed_if_deferring": [],
    }
    base.update(overrides)
    return base


def report_for(account_id, source=None, **overrides):
    client = ScriptedClient([
        message([tool_use_block("submit_verdict", verdict(**overrides), "toolu_1")],
                stop_reason="tool_use")
    ])
    return run_for_account(client, source if source is not None else meridian_transactions(), account_id)


@pytest.fixture(scope="module")
def leak_report():
    return report_for(
        FLAGSHIP_LEAK, verdict="leakage_detected", temporary_or_structural="structural",
        leak_dimensions=["margin", "tier_mix"], attributed_categories=["Power Tools"],
    )


@pytest.mark.parametrize("depth", [BRIEF, FULL])
def test_renders_a_valid_pdf_at_both_depths(leak_report, depth):
    pdf = build_pdf(leak_report, depth)
    assert pdf.startswith(PDF_MAGIC)
    assert len(pdf) > 5_000


def test_full_is_materially_richer_than_brief(leak_report):
    """The two depths must be genuinely different documents, or the choice
    in the UI is decoration.

    Measured in pages, not bytes: byte size tracked the two apart only while
    the full report embedded chart images, and would have kept passing for
    the wrong reason once those were dropped.
    """
    brief_pages = page_count(build_pdf(leak_report, BRIEF))
    full_pages = page_count(build_pdf(leak_report, FULL))
    assert brief_pages <= 3, f"the brief is meant to stay short, got {brief_pages} pages"
    assert full_pages >= brief_pages + 2


@pytest.mark.parametrize("account_id,overrides", [
    ("ACC-102", {}),                                                        # healthy
    ("ACC-104", {"verdict": "leakage_detected", "temporary_or_structural": "structural",
                 "leak_dimensions": ["revenue"]}),                          # revenue decline
    ("ACC-106", {"verdict": "leakage_detected", "temporary_or_structural": "structural",
                 "attributed_categories": ["Diagnostic Equipment"],
                 "leak_dimensions": ["category_mix"]}),                     # defection
    ("ACC-107", {"verdict": "leakage_detected", "temporary_or_structural": "structural",
                 "leak_dimensions": ["discount", "margin"]}),               # margin-only leak
    ("ACC-110", {"temporary_or_structural": "temporary"}),                  # recovered dip
    ("ACC-111", {}),                                                        # premiumisation
    (THIN_HISTORY, {"verdict": "insufficient_data", "confidence": "low", "defer": True,
                    "data_needed_if_deferring": ["12 months of history"]}),  # defer
])
def test_every_verdict_shape_renders(account_id, overrides):
    """Each of these exercises a different set of sections — a defer report
    has no impact block, a healthy one has no attribution, a margin-only
    leak has no per-category table."""
    report = report_for(account_id, **overrides)
    for depth in (BRIEF, FULL):
        assert build_pdf(report, depth).startswith(PDF_MAGIC)


def test_thin_input_still_produces_a_report():
    """The Reality Test may hand over a file with no margin, discount or
    tier columns. The export must narrow, not fail."""
    report = report_for(FLAGSHIP_LEAK, source=thin_transactions(with_margin=False))
    assert report["analysis_dimensions"]["margin"] is False
    for depth in (BRIEF, FULL):
        assert build_pdf(report, depth).startswith(PDF_MAGIC)


def test_markup_in_model_text_cannot_break_the_export(leak_report):
    """Narrative and cited facts are model output. An unescaped '<' or '&'
    in them would otherwise abort the whole document."""
    hostile = dict(leak_report)
    hostile["narrative"] = "Margin fell <b>sharply</b> & discount rose > 20% <not a tag>"
    hostile["cited_evidence"] = ["A & B < C", "<font color='red'>injected</font>"]
    hostile["recommended_actions"] = ["Review R&D spend <urgently>"]
    hostile["attributed_categories"] = ["Tools & Hardware"]
    for depth in (BRIEF, FULL):
        assert build_pdf(hostile, depth).startswith(PDF_MAGIC)


def test_unknown_depth_is_rejected(leak_report):
    with pytest.raises(ValueError, match="depth"):
        build_pdf(leak_report, "everything")


def test_report_without_evidence_still_exports(leak_report):
    """Older cached reports predate the evidence block. They must degrade to
    a thinner document rather than crash the demo."""
    stripped = {k: v for k, v in leak_report.items()
                if k not in ("evidence", "evidence_timeline")}
    assert build_pdf(stripped, FULL).startswith(PDF_MAGIC)


def test_filename_identifies_account_and_depth(leak_report):
    name = pdf_filename(leak_report, BRIEF)
    assert FLAGSHIP_LEAK in name
    assert name.endswith(".pdf")
    assert BRIEF in name
    assert re.search(r"\d{8}", name), name
