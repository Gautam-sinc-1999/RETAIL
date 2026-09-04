"""
Stage 7: Report Assembly — merges Stage 4's narrative with Stage 5/6's
rupee figures and priority bucket into the final output. Purely
templating/assembly, deterministic, no new LLM call — the narrative text
was already generated once, in Stage 4.

The report carries the full evidence pack and a derived evidence timeline
alongside the verdict. Without them the report is a conclusion with no way
back to the facts: the PDF export, and anything else that has to justify a
verdict rather than merely state it, would have no dates, no series and no
thresholds to show. Keeping them on the report also means a cached report
in demo_cache/ is as explainable offline as a live run is.
"""

from __future__ import annotations

from .timeline import build_timeline


def assemble_report(account_id: str, evidence_pack: dict, verdict: dict, impact: dict, priority: dict) -> dict:
    return {
        "account_id": account_id,
        "account_name": evidence_pack.get("account_name"),
        "verdict": verdict["verdict"],
        "temporary_or_structural": verdict["temporary_or_structural"],
        # Which dimension the value is leaving through (revenue, margin, mix,
        # discount, order_pattern). Two accounts can share a verdict of
        # "structural" and need completely different interventions; without
        # this the report cannot say which.
        "leak_dimensions": verdict.get("leak_dimensions", []),
        "confidence": verdict["confidence"],
        "defer": verdict["defer"],
        "narrative": verdict["narrative"],
        "attributed_categories": verdict["attributed_categories"],
        "cited_evidence": verdict["cited_facts"],
        "recommended_actions": verdict["recommended_actions"],
        "data_needed_if_deferring": verdict["data_needed_if_deferring"],
        "financial_impact": impact,
        "prioritization": priority,
        "data_sufficiency": evidence_pack["data_sufficiency"],
        "analysis_dimensions": evidence_pack.get("analysis_dimensions"),
        # The deterministic case behind the verdict, kept with it: a dated
        # event log for reading, and the pack itself so every figure in any
        # rendering is traceable to the stage that computed it.
        "evidence_timeline": build_timeline(evidence_pack),
        "evidence": evidence_pack,
    }
