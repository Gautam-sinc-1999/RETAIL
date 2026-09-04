"""
Demo UI (Stage 8 surface, per plan.md): upload a transaction CSV, run the
full 7-stage pipeline, and render the verdict, evidence, financial impact,
and prioritization. Two modes:

  - "Analyze a CSV": the real pipeline, including a live Stage 4 LLM call.
    Deterministic results (Stages 1-3) are shown even if Stage 4 fails, so
    a live-failure degrades gracefully instead of blanking the whole page.
  - "View offline demo": loads a pre-computed cached report from
    demo_cache/ — the live-failure fallback required by plan.md Phase 4
    ("offline fallback plan (cached run of a known archetype)"). Never
    silently substituted for a real uploaded file's failed run — the user
    picks it explicitly.

No hard-coded absolute paths — everything is relative to this file or
comes from the uploaded file object.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.agent import DEFAULT_MODEL, AgentError, pack_for_prompt
from pipeline.evidence import build_evidence_pack
from pipeline.ingest import IngestionError, account_sufficiency, ingest
from pipeline.impact import compute_impact
from pipeline.prioritize import prioritize
from pipeline.report import assemble_report
from pipeline.timeline import PRESENCE_ONLY_DIMENSIONS, READS_AS_ORDER
from validation.answer_key import (
    AnswerKeyUnavailable,
    load_answer_key,
    score_error,
    score_report,
    summarize,
)

DEMO_CACHE_DIR = REPO_ROOT / "demo_cache"
MERIDIAN_TRANSACTIONS = REPO_ROOT / "data" / "meridian" / "transactions.csv"

st.set_page_config(page_title="Revenue Leakage Investigator", layout="wide")


def get_live_client(provider: str, groq_model: str | None = None):
    """Construct the real LLM client for the chosen provider. Import is
    local so the app still runs (in offline-demo mode) even if a package
    or credentials aren't available.

    Groq is a temporary stand-in for Claude (see pipeline/groq_client.py)
    while the team doesn't yet have Anthropic API access — swapping back
    is just picking "Anthropic" in the sidebar once a key exists; nothing
    else in the pipeline is provider-aware."""
    if provider == "Groq":
        from pipeline.groq_client import GroqShimClient
        return GroqShimClient(model_override=groq_model)
    import anthropic
    return anthropic.Anthropic()


def run_agent_with_error_handling(client, df, account_id, evidence_pack, model, provider):
    from pipeline.agent import investigate

    error_module = __import__("groq") if provider == "Groq" else __import__("anthropic")
    provider_name = "Groq" if provider == "Groq" else "Anthropic"
    key_hint = "GROQ_API_KEY" if provider == "Groq" else "ANTHROPIC_API_KEY"

    try:
        return investigate(client, df, account_id, evidence_pack, model=model), None
    except error_module.AuthenticationError:
        return None, f"No valid {provider_name} API credentials — set {key_hint}, or use 'View offline demo' instead."
    except error_module.RateLimitError:
        return None, f"Rate limited by the {provider_name} API. Wait a moment and retry, or use 'View offline demo'."
    except error_module.APIConnectionError:
        return None, f"Could not reach the {provider_name} API (network issue). Use 'View offline demo' if this persists."
    except error_module.APIStatusError as e:
        return None, f"{provider_name} API error ({e.status_code}): {e.message}"
    except AgentError as e:
        return None, f"Agent did not reach a verdict: {e}"
    except Exception as e:
        return None, f"Unexpected error during investigation: {e}"


@st.cache_data(show_spinner=False)
def build_pdf_cached(report_json: str, depth: str) -> bytes:
    """Render a report to PDF, memoised on (report, depth).

    Keyed by the serialized report rather than the dict so Streamlit can
    hash it, and so that re-rendering the page — which happens on every
    widget interaction — does not rebuild the document each time.
    """
    from reporting.pdf import build_pdf

    return build_pdf(json.loads(report_json), depth)


PDF_DEPTH_LABELS = {
    "Full dossier": "full",
    "Executive brief": "brief",
}


def render_pdf_download(report: dict):
    """Depth picker plus a download button for the PDF export."""
    st.subheader("Download report")
    col1, col2 = st.columns([2, 3])
    label = col1.radio(
        "Depth", list(PDF_DEPTH_LABELS), horizontal=False,
        help=(
            "Executive brief: the verdict, the money, the dated timeline and the actions "
            "(~2 pages). Full dossier: adds per-dimension evidence with thresholds, the "
            "alternative explanations that were ruled out, product-level attribution, and a "
            "provenance appendix carrying every monthly series."
        ),
    )
    depth = PDF_DEPTH_LABELS[label]

    try:
        with st.spinner("Building PDF..."):
            from reporting.pdf import pdf_filename

            pdf_bytes = build_pdf_cached(json.dumps(report), depth)
    except Exception as e:
        # The PDF is a convenience on top of a report the page has already
        # rendered in full. A failure here must not blank the analysis.
        st.warning(f"Could not build the PDF: {e}. The report above is unaffected.")
        return

    col2.download_button(
        f"Download {label.lower()} (PDF)",
        data=pdf_bytes,
        file_name=pdf_filename(report, depth),
        mime="application/pdf",
        type="primary",
    )
    col2.caption(f"{len(pdf_bytes) / 1024:.0f} KB")


def render_timeline(report: dict):
    """The dated evidence log, as shown in the PDF."""
    timeline = report.get("evidence_timeline") or []
    if not timeline:
        return

    st.subheader("What changed, and when")
    st.caption(
        "Every row is a fact computed by the deterministic stages, dated to the month it was "
        "observed and marked for how it bears on the verdict. \"Checked\" rows are innocent "
        "explanations that were tested and did not account for what was found."
    )

    icons = {"concern": "🔴", "reassuring": "🟢", "checked": "🔍", "context": "⚪"}
    counts = {kind: sum(1 for e in timeline if e["reads_as"] == kind) for kind in READS_AS_ORDER}
    st.caption(" · ".join(
        f"{icons[kind]} {counts[kind]} {kind}" for kind in READS_AS_ORDER if counts[kind]
    ))

    st.dataframe(
        pd.DataFrame([
            {
                "": icons.get(event["reads_as"], ""),
                "When": event["when"],
                "What happened": event["headline"],
                "Detail": event["detail"],
            }
            for event in timeline
        ]),
        width="stretch",
        hide_index=True,
    )


def render_verdict_badge(verdict: str, defer: bool):
    if defer or verdict == "insufficient_data":
        st.warning(f"**Verdict: {verdict.replace('_', ' ').title()}** — evidence was insufficient for a confident call.")
    elif verdict == "leakage_detected":
        st.error(f"**Verdict: {verdict.replace('_', ' ').title()}**")
    else:
        st.success(f"**Verdict: {verdict.replace('_', ' ').title()}**")


def render_report(report: dict, cached: bool = False, allow_pdf: bool = False):
    if cached:
        st.info(
            f"Cached offline demo run ({report.get('_cache_metadata', {}).get('archetype', 'unknown')} archetype) — "
            "not a live LLM call. Used when live API access isn't available."
        )

    col1, col2, col3 = st.columns(3)
    account_label = report["account_id"]
    if report.get("account_name"):
        account_label = f"{account_label} · {report['account_name']}"
    col1.metric("Account", account_label)
    col2.metric("Confidence", report["confidence"].title())
    col3.metric("Temporary / Structural", report["temporary_or_structural"].replace("_", " ").title())

    render_verdict_badge(report["verdict"], report["defer"])
    st.write(report["narrative"])

    if report["defer"] and report["data_needed_if_deferring"]:
        st.write("**What data would resolve this:**")
        for item in report["data_needed_if_deferring"]:
            st.write(f"- {item}")

    # Which dimension the value is leaving through. Two accounts can share a
    # "structural" verdict and need entirely different interventions.
    if report.get("leak_dimensions"):
        st.write("**Leaking through:** " + ", ".join(
            d.replace("_", " ") for d in report["leak_dimensions"]
        ))

    if report["attributed_categories"]:
        st.subheader("Attributed categories")
        st.write(", ".join(report["attributed_categories"]))
    elif report["verdict"] == "leakage_detected" and not report["defer"]:
        st.caption(
            "No single category attributed — the loss is not explained by one line "
            "(naming a scapegoat category would send the account team after the wrong thing)."
        )

    # A dimension that could not be measured must never read as "measured and
    # fine" — say plainly what this file did not let us look at. "returns" is
    # excluded: it flags whether return lines are PRESENT, not whether they
    # could be analysed, so an account with no credit notes was not a gap in
    # coverage (see timeline.PRESENCE_ONLY_DIMENSIONS).
    unavailable = [
        name for name, available in (report.get("analysis_dimensions") or {}).items()
        if not available and name not in PRESENCE_ONLY_DIMENSIONS
    ]
    if unavailable:
        st.caption(
            "Not analysed (columns absent from this file): "
            + ", ".join(n.replace("_", " ") for n in unavailable)
        )

    render_timeline(report)

    st.subheader("Cited evidence")
    for fact in report["cited_evidence"]:
        st.write(f"- {fact}")

    impact = report["financial_impact"]
    margin_impact = impact.get("margin_impact")
    account_impact = impact.get("account_level_revenue_impact")

    if impact["per_category"] or margin_impact or account_impact:
        st.subheader("Financial impact")

        rows = [c for c in impact["per_category"] if c["quantifiable"]]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        for c in impact["per_category"]:
            if not c["quantifiable"]:
                st.caption(f"{c['category']}: not quantifiable — {c['reason']}")

        if account_impact:
            st.write(
                f"**Whole-account decline:** ₹{account_impact['baseline_monthly_revenue']:,.0f}/month "
                f"→ ₹{account_impact['recent_monthly_revenue']:,.0f}/month"
            )
            st.caption(account_impact["basis"])

        # Revenue and margin are shown side by side and never added: margin is
        # a slice of revenue, and a leak can be entirely in one with nothing
        # in the other (flat revenue, collapsing margin).
        col1, col2 = st.columns(2)
        col1.metric("Monthly revenue at risk", f"₹{impact['total_monthly_revenue_at_risk']:,.0f}")
        if margin_impact:
            col2.metric(
                "Monthly gross margin at risk",
                f"₹{margin_impact['monthly_margin_at_risk']:,.0f}",
                f"{margin_impact['margin_pct_erosion_pp']}pp margin rate",
                delta_color="inverse",
            )
            st.caption(
                f"Margin rate {margin_impact['baseline_margin_pct'] * 100:.1f}% → "
                f"{margin_impact['recent_margin_pct'] * 100:.1f}%. {margin_impact['basis']}"
            )
        if impact.get("overall_severity_pct_of_baseline"):
            st.caption(
                f"Severity: {impact['overall_severity_pct_of_baseline'] * 100:.1f}% of baseline "
                "(the worse of the revenue and margin ratios — they overlap and are not summed)."
            )

    priority = report["prioritization"]
    st.subheader("Prioritization")
    st.write(f"**Priority: {priority['priority']}** — {priority.get('reason', '')}")
    if priority.get("churn_risk_projection"):
        crp = priority["churn_risk_projection"]
        st.write(
            f"If this trend persists: projected loss of ₹{crp['projected_12_month_loss_if_unaddressed']:,.0f} "
            f"over 12 months (₹{crp['monthly_run_rate_loss']:,.0f}/month)."
        )
        st.caption(crp["basis"])

    if report["recommended_actions"]:
        st.subheader("Recommended actions")
        for action in report["recommended_actions"]:
            st.write(f"- {action}")

    with st.expander("Data sufficiency"):
        st.json(report["data_sufficiency"])

    if allow_pdf:
        render_pdf_download(report)


st.title("Revenue Leakage Investigator")
st.caption("Detect → Investigate → Attribute → Prioritise — Quessathon Retail challenge")

mode = st.sidebar.radio(
    "Mode", ["Analyze a CSV", "Answer Key validation", "View offline demo"]
)

provider = st.sidebar.radio(
    "LLM provider", ["Groq", "Anthropic"],
    help="Groq is a temporary stand-in while the team doesn't have Claude API access yet "
         "(see pipeline/groq_client.py). Switch to Anthropic once a key is available — "
         "nothing else in the pipeline needs to change.",
)
if provider == "Groq":
    from pipeline.groq_client import DEFAULT_GROQ_MODEL
    model = st.sidebar.text_input("Groq model", value=DEFAULT_GROQ_MODEL)
else:
    model = st.sidebar.text_input("Model", value=DEFAULT_MODEL)

if mode == "Answer Key validation":
    st.subheader("Answer Key validation")
    st.caption(
        "Runs the live agent over the reference dataset and compares each verdict against the "
        "workbook's Answer Key tab. The key is used only to score the output here — it is never "
        "shown to the agent."
    )

    try:
        key = load_answer_key()
    except AnswerKeyUnavailable as e:
        st.error(str(e))
        st.stop()

    if not MERIDIAN_TRANSACTIONS.exists():
        st.error(
            f"Reference transactions not found at {MERIDIAN_TRANSACTIONS}. "
            "Run `python scripts/prepare_dataset.py` to extract them from the workbook."
        )
        st.stop()

    validation_df, _ = ingest(str(MERIDIAN_TRANSACTIONS))
    all_accounts = sorted(validation_df["account_id"].unique())

    selected = st.multiselect(
        "Accounts to validate",
        all_accounts,
        default=all_accounts,
        format_func=lambda a: f"{a} — {key.get(a, {}).get('account_name', '')}",
    )
    st.caption(f"{len(selected)} account(s) selected — one live LLM call each.")

    if st.button("Run validation", type="primary", disabled=not selected):
        try:
            client = get_live_client(provider, groq_model=model if provider == "Groq" else None)
        except Exception as e:
            key_hint = "GROQ_API_KEY" if provider == "Groq" else "ANTHROPIC_API_KEY"
            st.error(f"Could not initialize the {provider} client: {e}. Set {key_hint}.")
            st.stop()

        results = []
        progress = st.progress(0.0)
        status = st.empty()
        for i, account_id in enumerate(selected, start=1):
            status.write(f"Investigating {account_id} ({i}/{len(selected)})…")
            try:
                pack = build_evidence_pack(validation_df, account_id)
                verdict, error = run_agent_with_error_handling(
                    client, validation_df, account_id, pack, model, provider
                )
                if error:
                    results.append(score_error(account_id, key[account_id], error))
                else:
                    impact = compute_impact(pack, verdict)
                    priority = prioritize(impact, verdict)
                    report = assemble_report(account_id, pack, verdict, impact, priority)
                    results.append(score_report(report, key[account_id]))
            except Exception as e:  # a single bad account must not lose the whole run
                results.append(score_error(account_id, key[account_id], str(e)))
            progress.progress(i / len(selected))
        status.empty()
        st.session_state["validation_results"] = results

    results = st.session_state.get("validation_results")
    if results:
        summary = summarize(results)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{summary['accuracy'] * 100:.0f}%",
                    f"{summary['matched']}/{summary['total']} accounts")
        # Broken out by expected outcome because one overall number cannot
        # distinguish a discriminating agent from one that flags everything.
        for column, outcome in zip((col2, col3, col4), ("FLAG", "NO FLAG", "DEFER")):
            bucket = summary["by_expected_outcome"].get(outcome)
            if bucket:
                column.metric(
                    f"Expected {outcome}", f"{bucket['accuracy'] * 100:.0f}%",
                    f"{bucket['matched']}/{bucket['total']}",
                )

        if summary["errors"]:
            st.warning(f"{summary['errors']} account(s) failed to produce a verdict — counted as misses.")

        table = pd.DataFrame([
            {
                "": "✅" if r["outcome_match"] else "❌",
                "Account": r["account_id"],
                "Name": r["account_name"],
                "Archetype": r["archetype"],
                "Expected": r["expected_verdict_text"],
                "Agent said": r["actual_outcome"],
                "Exp. confidence": r["expected_confidence"],
                "Confidence": r["actual_confidence"],
                "Leak dimensions": ", ".join(r["leak_dimensions"]),
                "Categories": ", ".join(r["attributed_categories"]),
                "Priority": r["priority"],
            }
            for r in results
        ])
        st.dataframe(table, width="stretch", hide_index=True)

        if summary["mismatches"]:
            st.subheader("Mismatches")
            for line in summary["mismatches"]:
                st.write(f"- {line}")

        st.subheader("Per-account detail")
        for r in results:
            icon = "✅" if r["outcome_match"] else "❌"
            with st.expander(f"{icon} {r['account_id']} — {r['account_name']} ({r['archetype']})"):
                st.write(f"**What this account tests:** {r['what_it_tests']}")
                st.write(f"**What is really happening:** {r['what_is_really_happening']}")
                st.write(
                    f"**Expected:** {r['expected_verdict_text']} "
                    f"({r['expected_confidence']}, {r['expected_leak_type']})"
                )
                if r.get("error"):
                    st.error(f"Run failed: {r['error']}")
                else:
                    st.write(
                        f"**Agent said:** {r['actual_outcome']} "
                        f"({r['actual_confidence']} confidence, {r['actual_temporary_or_structural']})"
                    )
                    st.write(r["narrative"])
                    if r["cited_evidence"]:
                        st.write("**Cited evidence:**")
                        for fact in r["cited_evidence"]:
                            st.write(f"- {fact}")

        st.download_button(
            "Download scorecard (JSON)",
            data=json.dumps({"summary": summary, "results": results}, indent=2),
            file_name="answer_key_scorecard.json",
            mime="application/json",
        )
    else:
        st.info("Select accounts and click **Run validation** to score the agent against the Answer Key.")

elif mode == "View offline demo":
    cache_files = sorted(DEMO_CACHE_DIR.glob("*.json")) if DEMO_CACHE_DIR.exists() else []
    if not cache_files:
        st.error(f"No cached demo reports found in {DEMO_CACHE_DIR}. Run scripts/generate_demo_cache.py first.")
    else:
        choice = st.selectbox("Cached archetype", [f.stem for f in cache_files])
        report = json.loads((DEMO_CACHE_DIR / f"{choice}.json").read_text())
        render_report(report, cached=True)

else:
    uploaded = st.file_uploader("Upload a transaction CSV", type=["csv"])
    if uploaded is not None:
        try:
            df, ingestion_report = ingest(uploaded)
        except IngestionError as e:
            st.error(f"Could not process this file: {e}")
            st.stop()

        with st.expander("Ingestion report"):
            st.json(ingestion_report)

        account_ids = sorted(df["account_id"].unique())
        account_id = st.selectbox("Account to investigate", account_ids)

        sufficiency = account_sufficiency(df).get(account_id)
        if sufficiency:
            st.caption(
                f"Data sufficiency for {account_id}: **{sufficiency['label']}** "
                f"({sufficiency['history_months']} months, {sufficiency['order_count']} orders, "
                f"{sufficiency['category_count']} categories)"
            )

        if st.button("Run investigation", type="primary"):
            # The result is stashed rather than rendered inline: every widget
            # below it (the PDF depth picker, the download button itself)
            # triggers a Streamlit rerun, at which point the button reads
            # False and an inline render would blank the page mid-demo.
            st.session_state.pop("live_run", None)

            with st.spinner("Running deterministic analysis (Stages 1-3)..."):
                evidence_pack = build_evidence_pack(df, account_id)

            with st.spinner(f"Running investigation agent (Stage 4, via {provider})..."):
                try:
                    client = get_live_client(provider, groq_model=model if provider == "Groq" else None)
                except Exception as e:
                    key_hint = "GROQ_API_KEY" if provider == "Groq" else "ANTHROPIC_API_KEY"
                    st.error(
                        f"Could not initialize the {provider} client: {e}. "
                        f"Set {key_hint}, or switch to 'View offline demo' in the sidebar."
                    )
                    st.stop()

                verdict, error = run_agent_with_error_handling(client, df, account_id, evidence_pack, model, provider)

            run = {"account_id": account_id, "evidence_pack": evidence_pack, "error": error}
            if not error:
                impact = compute_impact(evidence_pack, verdict)
                priority = prioritize(impact, verdict)
                run["report"] = assemble_report(account_id, evidence_pack, verdict, impact, priority)
            st.session_state["live_run"] = run

        run = st.session_state.get("live_run")
        # A stored run belongs to the account it was run for; switching the
        # selectbox must not leave the previous account's verdict on screen.
        if run and run["account_id"] == account_id:
            with st.expander("Evidence pack (what the agent sees)"):
                # Exactly what Stage 4 receives — the presentation-only keys
                # the PDF uses are stripped, so this expander cannot drift
                # from the real prompt payload.
                st.json(pack_for_prompt(run["evidence_pack"]))

            if run.get("error"):
                st.error(run["error"])
                st.info(
                    "Deterministic evidence (Stages 1-3) above is still valid and traceable — "
                    "only the LLM reasoning step failed. Retry, or use 'View offline demo' in the sidebar."
                )
            else:
                render_report(run["report"], allow_pdf=True)
    else:
        st.write("Upload a CSV to begin, or switch to 'View offline demo' in the sidebar.")
