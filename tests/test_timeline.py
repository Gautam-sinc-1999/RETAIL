"""
Tests for the evidence timeline (Stage 7's dated event log).

The timeline restates deterministic facts, so what matters is not that it
produces events but that it produces the RIGHT ones with the right dates
and the right bearing on the verdict. Each test below pins a discriminating
behaviour against the real workbook data.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from datasets import DEFECTED_ACCOUNT, DEFECTED_CATEGORY, FLAGSHIP_LEAK, THIN_HISTORY, meridian_transactions
from pipeline.agent import pack_for_prompt
from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest
from pipeline.timeline import build_timeline, timeline_counts


@pytest.fixture(scope="module")
def df():
    frame, _ = ingest(meridian_transactions())
    return frame


def pack_for(df, account_id):
    return build_evidence_pack(df, account_id)


def headlines(timeline):
    return [event["headline"] for event in timeline]


def find(timeline, fragment):
    return [e for e in timeline if fragment.lower() in e["headline"].lower()]


def test_presentation_block_never_reaches_the_model(df):
    """The whole point of the underscore convention: the report may grow
    without changing a single byte of the Stage 4 prompt."""
    pack = pack_for(df, FLAGSHIP_LEAK)
    assert "_presentation" in pack

    seen_by_model = pack_for_prompt(pack)
    assert "_presentation" not in seen_by_model
    assert not [key for key in seen_by_model if key.startswith("_")]
    # Everything else must survive — stripping is not filtering.
    assert set(seen_by_model) == {k for k in pack if not k.startswith("_")}


def test_report_only_data_stays_out_of_the_prompt(df):
    """Chart series and rep-change dates exist for the PDF alone. If any of
    them leaks into the prompt, Stage 4's tuned input has silently changed
    and the answer-key scores no longer describe the shipped agent."""
    report_only = ("category_monthly_revenue", "account_manager_tenures",
                   "monthly_avg_discount_series", "monthly_high_tier_share_series")
    for account_id in sorted(df["account_id"].unique()):
        prompt_text = json.dumps(pack_for_prompt(pack_for(df, account_id)))
        for key in report_only:
            assert key not in prompt_text, f"{key} leaked into {account_id}'s prompt"


def test_flagship_leak_timeline_dates_the_margin_and_mix_collapse(df):
    """ACC-101 holds revenue flat while margin and tier mix collapse. The
    timeline must date both, and must not report the flat revenue as a
    concern."""
    timeline = build_timeline(pack_for(df, FLAGSHIP_LEAK))

    margin = find(timeline, "margin rate erodes")
    assert margin, headlines(timeline)
    assert margin[0]["reads_as"] == "concern"
    # Dated to the month the rate left its baseline band, not to the window.
    assert margin[0]["month"].startswith("2026-")
    assert "-12.3pp" in margin[0]["detail"]

    downgrade = find(timeline, "product mix downgrades")
    assert downgrade and downgrade[0]["reads_as"] == "concern"
    assert "-43.8pp" in downgrade[0]["detail"]

    revenue = find(timeline, "revenue is stable")
    assert revenue and revenue[0]["reads_as"] == "reassuring"


def test_timeline_is_chronological_with_undated_events_last(df):
    timeline = build_timeline(pack_for(df, FLAGSHIP_LEAK))
    dated = [e["month"] for e in timeline if e["month"]]
    assert dated == sorted(dated)
    # Once an undated event appears, no dated event may follow it.
    seen_undated = False
    for event in timeline:
        if event["month"] is None:
            seen_undated = True
        elif seen_undated:
            pytest.fail("a dated event sorted after an undated one")


def test_defection_is_dated_to_the_last_month_purchased(df):
    """'Stopped after 2025-10' is actionable; 'ten months at zero' makes the
    reader do the subtraction."""
    timeline = build_timeline(pack_for(df, DEFECTED_ACCOUNT))
    stops = find(timeline, f"{DEFECTED_CATEGORY} stops completely")
    assert stops, headlines(timeline)

    event = stops[0]
    assert event["reads_as"] == "concern"
    assert event["month"] is not None
    series = pack_for(df, DEFECTED_ACCOUNT)["_presentation"]["category_monthly_revenue"][DEFECTED_CATEGORY]
    # The dated month is the last one with revenue; every later month is zero.
    assert series[event["month"]] > 0
    assert all(value == 0 for month, value in series.items() if month > event["month"])


def test_thin_history_reports_what_it_could_not_establish(df):
    """The DEFER account's timeline is mostly absences. Each unmeasurable
    dimension must appear as its own concern rather than being silently
    omitted, which would read as an all-clear."""
    timeline = build_timeline(pack_for(df, THIN_HISTORY))
    text = " ".join(headlines(timeline)).lower()

    # Margin, discount and tier mix share one cause (no baseline window) and
    # are reported in a single merged row rather than three identical ones.
    for expected in ("margin, discount and tier mix could not be compared",
                     "seasonality could not be checked",
                     "revenue direction could not be established"):
        assert expected in text, headlines(timeline)

    limits = find(timeline, "limits on what this history can establish")
    assert limits and limits[0]["reads_as"] == "concern"
    assert "history too short for baseline" in limits[0]["detail"]

    assert timeline_counts(timeline)["reassuring"] == 0


def test_healthy_account_timeline_is_dominated_by_reassurance(df):
    """The discrimination the product exists for, viewed through the
    timeline: a clean account must not accumulate concerns."""
    counts = timeline_counts(build_timeline(pack_for(df, "ACC-102")))
    assert counts["reassuring"] > counts["concern"]


def test_premiumisation_reads_as_good_news_not_a_mix_change(df):
    """ACC-111 moves as much tier share as the flagship leak, in the other
    direction. Direction, not magnitude, decides how it reads."""
    timeline = build_timeline(pack_for(df, "ACC-111"))
    upmarket = find(timeline, "product mix moves upmarket")
    assert upmarket, headlines(timeline)
    assert upmarket[0]["reads_as"] == "reassuring"
    assert not find(timeline, "product mix downgrades")


def test_bulk_month_is_checked_not_counted_as_a_decline(df):
    timeline = build_timeline(pack_for(df, "ACC-114"))
    bulk = find(timeline, "bulk month")
    assert bulk, headlines(timeline)
    assert bulk[0]["reads_as"] == "checked"
    assert bulk[0]["month"] is not None


def test_skipped_month_is_reported_as_a_data_gap(df):
    timeline = build_timeline(pack_for(df, "ACC-117"))
    gaps = find(timeline, "no orders at all")
    assert gaps, headlines(timeline)
    assert gaps[0]["reads_as"] == "checked"
    assert "not a month of zero trading" in gaps[0]["detail"]


def test_returns_are_declared_as_already_netted(df):
    timeline = build_timeline(pack_for(df, "ACC-116"))
    returns = find(timeline, "return/credit line")
    assert returns, headlines(timeline)
    assert returns[0]["reads_as"] == "checked"
    assert "not leakage" in returns[0]["detail"]


def test_absent_returns_are_not_reported_as_missing_coverage(df):
    """`analysis_dimensions["returns"]` flags PRESENCE of credit notes, not
    the ability to analyse them — a clean account must not be told a
    dimension was unanalysable."""
    timeline = build_timeline(pack_for(df, FLAGSHIP_LEAK))
    coverage = find(timeline, "could not be analysed")
    assert not coverage, [e["detail"] for e in coverage]


def test_thresholds_live_in_method_not_in_the_business_detail(df):
    """`detail` is what an account owner reads. Threshold arithmetic belongs
    in `method`, which only the full dossier prints — a Rs 23k/month problem
    should not be phrased as a sentence about percentage points."""
    for account_id in sorted(df["account_id"].unique()):
        for event in build_timeline(pack_for(df, account_id)):
            assert "threshold" not in (event["detail"] or "").lower(), event
            assert "requires both" not in (event["detail"] or "").lower(), event


def test_a_leaking_account_has_thresholds_available_for_review(df):
    """Dropping the thresholds from `detail` must not lose them — the full
    report still has to be able to defend every call."""
    timeline = build_timeline(pack_for(df, "ACC-107"))
    methods = " ".join(e["method"] or "" for e in timeline).lower()
    assert "threshold" in methods
    assert any("creep threshold" in (e["method"] or "").lower() for e in timeline)


def test_notable_marks_movement_not_badness(df):
    """A healthy account that is trading up has a finding worth reporting.
    If only concerns were notable, the brief would show a minor product stop
    and bury the premiumisation that explains the whole account."""
    timeline = build_timeline(pack_for(df, "ACC-111"))
    notable = {e["headline"] for e in timeline if e["notable"]}
    assert any("moves upmarket" in h for h in notable), notable

    # "Nothing changed" rows stay out of the shortlist.
    quiet = {e["headline"] for e in timeline if not e["notable"]}
    assert any("is stable" in h or "unchanged" in h for h in quiet), quiet

    # Every concern is notable by definition.
    for event in timeline:
        if event["reads_as"] == "concern":
            assert event["notable"], event


def test_every_event_is_well_formed(df):
    """A malformed event would render as a blank table row in the PDF."""
    for account_id in sorted(df["account_id"].unique()):
        for event in build_timeline(pack_for(df, account_id)):
            assert event["headline"], account_id
            assert event["detail"], event
            assert event["reads_as"] in ("concern", "reassuring", "checked", "context")
            assert event["when"], event
