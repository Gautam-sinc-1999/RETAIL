"""
Tests for the PDF export.

These do not assert on visual layout — that is checked by eye. They pin the
things that would silently produce a wrong or broken document: that every
verdict shape renders at both depths, that hostile text in a narrative
cannot abort the export, that a thin file still produces a report, and that
the export never invents a number the pipeline did not compute.
"""

import base64
import re
import sys
import zlib
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


def pdf_text(pdf: bytes) -> str:
    """Visible text of a PDF.

    reportlab Flate-compresses page content streams, so the rendered words
    are not present in the raw bytes — asserting against `pdf.decode()`
    silently matches nothing and passes for the wrong reason. Decompress
    each stream and pull the literals out of the text operators.
    """
    text = []
    # The closing newline is optional: reportlab writes "...~>endstream".
    for raw in re.findall(rb"stream\r?\n(.*?)(?:\r?\n)?endstream", pdf, re.S):
        content = raw.strip()
        # reportlab writes /Filter [ /ASCII85Decode /FlateDecode ], so both
        # layers have to come off before any words are visible.
        try:
            if content.endswith(b"~>"):
                content = base64.a85decode(content[:-2])
            content = zlib.decompress(content)
        except Exception:
            continue
        for literal in re.findall(rb"\((?:[^()\\]|\\.)*\)", content):
            text.append(_unescape_pdf_literal(literal[1:-1]))
    return b" ".join(text).decode("latin-1")


# PDF string literals escape non-printables as octal (\177 is the notdef box
# reportlab substitutes for an unencodable character). Without decoding
# these, a search for "\x7f" finds nothing and a notdef test passes while
# the document is visibly broken.
_PDF_ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b",
                b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}


def _unescape_pdf_literal(raw: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(raw):
        if raw[i:i + 1] != b"\\":
            out += raw[i:i + 1]
            i += 1
            continue
        nxt = raw[i + 1:i + 2]
        octal = re.match(rb"[0-7]{1,3}", raw[i + 1:i + 4])
        if octal:
            out.append(int(octal.group(), 8) & 0xFF)
            i += 1 + len(octal.group())
        else:
            out += _PDF_ESCAPES.get(nxt, nxt)
            i += 2
    return bytes(out)


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


def test_brief_stays_to_the_business_facts(leak_report):
    """The brief must not carry the agent's raw citations or the threshold
    methodology — both are reviewer material, and both are what pushed the
    earlier brief onto a second page."""
    brief = pdf_text(build_pdf(leak_report, BRIEF))
    full = pdf_text(build_pdf(leak_report, FULL))

    for reviewer_only in ("Facts the verdict rests on", "How this was judged"):
        assert reviewer_only not in brief, reviewer_only
        assert reviewer_only in full, reviewer_only


def test_brief_omits_a_coverage_section_when_nothing_is_limited(leak_report):
    """"Data sufficiency: sufficient" is a line saying nothing was wrong.
    It earns space only when something was."""
    assert leak_report["data_sufficiency"]["label"] == "sufficient"
    assert not leak_report["data_sufficiency"]["flags"]
    assert "Coverage and limitations" not in pdf_text(build_pdf(leak_report, BRIEF))


def test_brief_keeps_coverage_when_the_history_is_thin():
    """The inverse: on the deferral account the limitation IS the finding."""
    report = report_for(THIN_HISTORY, verdict="insufficient_data", confidence="low", defer=True,
                        data_needed_if_deferring=["12 months of history"])
    assert "Coverage and limitations" in pdf_text(build_pdf(report, BRIEF))


def test_unicode_in_model_text_never_renders_as_a_notdef_box(leak_report):
    """The built-in Helvetica encodes cp1252 only, and reportlab silently
    substitutes a black box (0x7F) for anything outside it.

    This is not hypothetical: a live ACC-109 report printed "Re<box>run"
    because the model wrote a non-breaking hyphen. Characters with an ASCII
    equivalent must be mapped, and anything left must be dropped rather than
    boxed.
    """
    report = dict(leak_report)
    report["narrative"] = "Re‑run the check – margin → 12%, discount ≥ 20%."
    report["recommended_actions"] = ["Review high‑value lines ‘now’ — ₹5,000/mo"]
    report["cited_evidence"] = ["Tier mix → down", "emoji \U0001f600 should vanish"]

    for depth in (BRIEF, FULL):
        pdf = build_pdf(report, depth)
        # The tell is a font switch, not a character code: reportlab renders
        # an unencodable glyph by falling back to ZapfDingbats, where the
        # substituted letter draws as a filled square. A clean document never
        # references that font. (Checking for \x7f does NOT work — that is
        # also the code for the ordinary bullet, which renders correctly.)
        assert b"ZapfDingbats" not in pdf, f"fallback font used in the {depth} report"
        assert b"Symbol" not in pdf, f"fallback font used in the {depth} report"

        squeezed = pdf_text(pdf).replace(" ", "")
        assert "Re-runthecheck" in squeezed
        assert "margin->12%" in squeezed
        assert "discount>=20%" in squeezed
        assert "Rs5,000/mo" in squeezed


def test_the_no_baseline_dimensions_are_reported_once(leak_report):
    """On a too-new account, margin/discount/tier mix each report the same
    single cause. Three identical rows read as three findings."""
    report = report_for(THIN_HISTORY, verdict="insufficient_data", confidence="low", defer=True,
                        data_needed_if_deferring=["12 months of history"])
    headlines = [e["headline"] for e in report["evidence_timeline"]]

    merged = [h for h in headlines if "could not be compared" in h]
    assert len(merged) == 1, headlines
    for dimension in ("margin", "discount", "tier mix"):
        assert dimension in merged[0].lower()


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
