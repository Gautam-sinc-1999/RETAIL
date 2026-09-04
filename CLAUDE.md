# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Quessathon "Retail: Where Did the Revenue Go?" challenge entry. An agent that investigates one B2B account's transaction history and decides whether it shows healthy behaviour, temporary variation, or structural revenue leakage — including deferring when the evidence can't support a confident call.

[plan.md](plan.md) is the living project document: architecture rationale, phase checklist, standing decisions, and current status. Read it before making architectural changes, and update it when decisions change. (It references `data.md` and `meeting_notes.md`, which are not in this repo.)

## Commands

`venv/` is not checked in and there is no pytest config; tests rely on `sys.path.insert` in each test file, so run pytest from the repo root.

```bash
python -m venv venv
venv/Scripts/pip install -r requirements-dev.txt   # venv/bin/pip on POSIX
cp .env.example .env                               # fill in GROQ_API_KEY (or ANTHROPIC_API_KEY)

venv/Scripts/python scripts/prepare_dataset.py     # workbook -> data/meridian/*.csv (run first)
venv/Scripts/streamlit run app.py                  # demo UI
venv/Scripts/python -m pytest tests/ -v            # all 164 tests
venv/Scripts/python -m pytest tests/test_agent.py::test_happy_path_one_drilldown_then_submit -v   # one test

venv/Scripts/python scripts/validate_answer_key.py            # score the live agent vs the Answer Key
venv/Scripts/python scripts/generate_demo_cache.py            # refresh demo_cache/
```

`validate_answer_key.py` takes `--provider`, `--model`, `--accounts`, and `--out`, and exits non-zero on any mismatch.

## The dataset is the specification

`Quessathon_Revenue_Leakage_Dataset.xlsx` is the official input and the thing every threshold in this repo is calibrated against. [scripts/prepare_dataset.py](scripts/prepare_dataset.py) flattens its six tabs into `data/meridian/`; the pipeline only ever reads `transactions.csv` from there.

18 accounts, 551 orders, 3,289 lines, Sep 2024 – Aug 2026, 17 columns. The design consequences:

- **Only 6 of 18 accounts are genuine leaks.** "Always flag" is wrong two-thirds of the time. False-positive discipline is the product, and several accounts (seasonal dip, recovered dip, premiumisation, bulk order, returns, skipped month, 12% drift) exist purely to punish over-flagging.
- **Two of the six leaks are invisible in revenue.** ACC-101 holds revenue flat while high-tier share falls 44pp and margin falls 12pp; ACC-107 holds volume and mix steady while discount creeps 5%→25%. A revenue-only pipeline scores zero on both. This is why Stage 2/3 profile margin, discount, and tier mix as first-class dimensions rather than as extras.
- **ACC-109 has 5 months of history and must DEFER.** It's the rubric's heaviest single line.
- **The mess is deliberate**: 3 return/credit lines, one 2.8× bulk month, one skipped month, a mid-history rep change. Each has a matching handler in Stage 3's `data_quality` block.

`data/meridian/answer_key.json` is ground truth **for scoring only** — the workbook's READ ME says so explicitly, and `monthly_summary.csv` is likewise "a derived convenience, not an agent input". Nothing under `src/pipeline/` may read either.

## Architecture

Seven stages, run for one account at a time. [src/pipeline/orchestrate.py](src/pipeline/orchestrate.py) is the whole wiring in ~10 lines and is the fastest way to see the shape:

```
ingest → build_evidence_pack → investigate (LLM) → compute_impact → prioritize → assemble_report
```

| Stage | File | Nature |
| --- | --- | --- |
| 1 Ingest / validate / sufficiency | [ingest.py](src/pipeline/ingest.py) | deterministic |
| 2 Baseline engine | [baseline.py](src/pipeline/baseline.py) | deterministic |
| 3 Change detection | [detect.py](src/pipeline/detect.py), [changepoint.py](src/pipeline/changepoint.py) | deterministic |
| — Evidence pack assembly | [evidence.py](src/pipeline/evidence.py) | deterministic |
| 4 Investigation & attribution | [agent.py](src/pipeline/agent.py) | **the one LLM call** |
| 5 Rupee impact | [impact.py](src/pipeline/impact.py) | deterministic |
| 6 Prioritisation | [prioritize.py](src/pipeline/prioritize.py) | deterministic |
| 7 Report assembly | [report.py](src/pipeline/report.py), [timeline.py](src/pipeline/timeline.py) | deterministic |

[src/validation/answer_key.py](src/validation/answer_key.py) sits deliberately **outside** `src/pipeline/` — it scores finished reports against the workbook's Answer Key and must be unreachable from anything the agent touches. [src/reporting/](src/reporting/) (the PDF export) sits outside for the same reason: it reads finished reports and renders them, and never computes an analytical figure.

### The report carries its own evidence

`assemble_report` attaches the full evidence pack plus `evidence_timeline` — a dated, ordered event log built by [timeline.py](src/pipeline/timeline.py). Each event carries a month, the fact, and a `reads_as` of `concern` / `reassuring` / `checked` / `context`. That last field is what makes the output an argument rather than a list: a report showing only the concerns is a prosecution, and on a NO FLAG account the `reassuring` and `checked` rows *are* the entire finding. Without this the report is a verdict with no path back to the facts.

### `_presentation` — the report/prompt boundary

`build_evidence_pack` emits a `_presentation` block (monthly discount and tier-share series, per-category monthly revenue, rep-change dates). [agent.py](src/pipeline/agent.py) `pack_for_prompt` strips every underscore-prefixed key before serializing the pack into the Stage 4 message.

This exists so the report can be enriched without ever changing what the model reads. The prompt is tuned against all 18 reference accounts and the unit tests use scripted verdicts, so **growing the model's input is a regression the test suite cannot detect**. Put anything the report needs and the model does not behind that prefix; `tests/test_timeline.py::test_report_only_data_stays_out_of_the_prompt` fails if report-only data leaks into the prompt.

Note that `analysis_dimensions["returns"]` is a *presence* flag (are there credit notes?), not a capability flag like the rest. `timeline.PRESENCE_ONLY_DIMENSIONS` excludes it from "could not be analysed" messaging — without that, a clean account is told a dimension was unmeasurable.

### Six detection dimensions, not one

Stages 2 and 3 profile revenue, margin, discount, category mix, tier mix, and order pattern in parallel; Stage 4 weighs them. Each maps to an account in the dataset that is invisible in the others:

| Dimension | Detector status values | The account that needs it |
| --- | --- | --- |
| revenue | `material_decline` / `mild_drift` / `stable` / `growth` | ACC-104, ACC-112 vs ACC-118 |
| margin | `erosion_detected` / `stable` / `improvement_detected` | ACC-101, ACC-107 |
| discount | `creep_detected` / `stable` / `discipline_improved` | ACC-107 |
| tier_mix | `downgrade_detected` / `stable` / `premiumisation_detected` | ACC-101 vs ACC-111 |
| order_pattern | `fragmentation_detected` / `baskets_shrinking` / `stable` | ACC-108 |
| category | `defected` + `consecutive_months_at_zero` | ACC-106 |

Three rules that fall out of this and are easy to break:

- **Direction is the signal, not magnitude.** ACC-101 and ACC-111 both hold revenue flat with a large tier-mix shift; one is the flagship leak and the other is a healthy account trading up. Any detector that reports a mix change as a magnitude makes them identical.
- **`analysis_dimensions`** (from `ingest.analysis_dimensions`) states which dimensions the input file could actually support. It exists so "we could not measure margin" is never collapsed into "margin was fine". Preserve that distinction in any new detector.
- **Revenue at risk and margin at risk are never summed** — margin is a slice of revenue. `impact.py` reports both and takes the *worse of the two ratios* as severity, so a flat-revenue account bleeding a third of its gross margin doesn't rank Low.

### Invariants that shape the whole design

- **Exactly one LLM call site.** Stage 4 is the only judgement call that can't be reduced to a threshold. Do not add model calls for prioritisation, explanation, or narrative rewriting — narrative and recommended actions are generated once, in Stage 4. Consolidating four LLM stages into one was a deliberate decision (see plan.md).
- **The LLM never computes a number.** Every figure in any output traces back to deterministic code via the evidence pack or a drill-down tool result. Stage 5 sizes impact from `category_changes` facts, not from anything the model said.
- **The LLM never sees raw transactions** — only `build_evidence_pack`'s structured JSON.
- **Deferring is a feature, not a failure.** `defer=true` with low confidence and a populated `data_needed_if_deferring` is the designed behaviour when history is short or signals conflict; it scores higher than a forced verdict. Don't "improve" the agent toward decisiveness.
- **Every verdict carries confidence + cited evidence.** An output missing either is a bug.
- **Stage 1 must never crash on an unfamiliar CSV.** It either resolves the input to the canonical schema (synonym match, then fuzzy match, then derivation) or raises `IngestionError` naming exactly what's missing. This matters because the event's "Reality Test" feeds an unseen file live. Everything past `REQUIRED_FIELDS` is enrichment: present → an extra detection dimension lights up; absent → the analysis narrows and says so. It must never break.
- **The Answer Key never reaches the agent.** It scores finished reports, nothing more.

### The LLM client is injected, not constructed

`investigate(client, ...)` takes any object exposing `client.messages.create(model, max_tokens, system, tools, messages)` returning an Anthropic-shaped response (`.stop_reason`, `.content` blocks with `.type`/`.text`/`.name`/`.input`/`.id`). Three implementations exist:

- `anthropic.Anthropic()` — the real target.
- [groq_client.py](src/pipeline/groq_client.py) `GroqShimClient` — a temporary stand-in used because Claude API access isn't available yet. It translates Anthropic-shaped tools/messages to Groq's OpenAI-compatible format and back. It is deliberately isolated and swappable: nothing else in the pipeline is provider-aware, and deleting this file plus picking "Anthropic" in the UI sidebar is the entire swap. Do not let Groq-specific handling leak into `agent.py` or downstream stages.
- [tests/fakes.py](tests/fakes.py) `ScriptedClient` — returns pre-scripted responses in order, so the agent loop and all downstream stages are testable without an API.

### Noise handling is load-bearing

Accounts have few orders per month, so raw monthly revenue swings wildly. Several constants in [changepoint.py](src/pipeline/changepoint.py) exist to stop that noise from manufacturing verdicts, and each was tuned against a specific observed failure — read the docstrings before changing any of them:

- `winsorize` caps values at 1.5× the series median so one bulk order can't be smeared across adjacent points by the rolling sum.
- `rolling` (trailing 3-month sum) smooths every series before any threshold or change-point logic.
- `MIN_SEGMENT_MONTHS = 5` prevents a single abnormal month from defining a whole segment baseline.
- `RECENT_MONTHS = 6` — 3 was too few; a healthy account swung −59% from pure sampling variance.
- `normalize_medians_to_monthly_rate` **must** be called on any change-point computed from a rolled (summed) series before its medians reach anything rupee-denominated, or every figure is 3× inflated.
- The recovery check skips the first `ROLLING_WINDOW - 1` post-split points, which the trailing sum contaminates with pre-change months.
- `_seasonal_precedent` in [detect.py](src/pipeline/detect.py) only compares against months *strictly before* the decline, and requires 2+ prior occurrences of the same calendar month — roughly three years of history. On the 24-month dataset it therefore returns `insufficient_history` on the very account it was meant to resolve. `prior_year_echo` is the check 24 months *can* support ("did this same dip happen twelve months ago"), and is what actually catches ACC-103. Both are reported; neither replaces the other.
- Order behaviour uses window **means**, not medians. Basket width is a small integer, so its median moves in whole steps — a healthy control read as a flat −50% cliff before this change.
- `revenue_trend` winsorizes before fitting, or ACC-114's one bulk month reads as growth on a flat account.
- Thresholds in `detect.py` (`MARGIN_EROSION_PP`, `DISCOUNT_CREEP_PP`, `HIGH_TIER_SHIFT_PP`, the `MATERIAL_DECLINE_*` pair) sit an order of magnitude above the dataset's own noise floor and well below its real signals. Changing one means re-checking all 18 accounts, not one.

### Stage 4 prompt is tuned against archetypes

`SYSTEM_PROMPT` in [agent.py](src/pipeline/agent.py) encodes the temporary-vs-structural criteria explicitly: check every dimension not just revenue, direction-is-not-magnitude, rule out the innocent explanations (prior-year echo, recovered dip, data gap, bulk month, returns) before flagging, attribution honesty for diffuse declines, and the defer instruction. Each rule maps to an account that would otherwise be misclassified. Changing the prompt means re-running `scripts/validate_answer_key.py` across all 18 accounts — the unit tests use scripted verdicts and **cannot** catch a prompt regression.

## Testing

There is exactly one dataset: `data/meridian/`, the official workbook extracted. An earlier home-grown generator (`generate_data.py` + `synthetic_data/`) was removed once the workbook became the calibration target — a second dataset that disagreed with the spec was testing a different product.

Thin-input coverage comes from [tests/datasets.py](tests/datasets.py) instead: `thin_transactions()` derives a degraded file by *dropping columns* from the real data (no tier, discount, list price, unit cost; optionally no margin), so the same accounts and figures exercise the narrow path the Reality Test may hand over. It keeps the workbook's raw column names, so the synonym mapping stays exercised too.

`tests/` runs end-to-end with `ScriptedClient` standing in for the model, so it needs no API key:

- [test_evidence.py](tests/test_evidence.py) — the deterministic layer. One test per discriminating behaviour, plus a population check that the six FLAG accounts each trip a material signal and none of the eleven cleared accounts do.
- [test_orchestrate.py](tests/test_orchestrate.py) — wiring and rupee math for every verdict shape, including the margin-only and diffuse-decline cases and the double-count guard.
- [test_answer_key.py](tests/test_answer_key.py) — the scorecard logic: that a perfect run reaches 100% and "always flag" scores 6/18.
- [test_timeline.py](tests/test_timeline.py) — the dated event log, and the `_presentation` boundary that keeps the prompt fixed.
- [test_pdf_report.py](tests/test_pdf_report.py) — that every verdict shape renders at both depths, that model text containing markup cannot abort the export, and that a thin input narrows the document rather than breaking it. Depth difference is asserted in *pages*, not bytes: byte size only separated the two while charts were embedded.

None of it measures model judgement; that is `scripts/validate_answer_key.py`, which needs a live key.

## Demo UI

[app.py](app.py) has three modes:

- **Analyze a CSV** — the live pipeline. If Stage 4 fails it still shows the deterministic Stage 1–3 evidence rather than blanking the page. The finished run is held in `st.session_state["live_run"]` and rendered *outside* the button block: every widget below it (the PDF depth picker, the download button) triggers a rerun, and an inline render would blank the page mid-demo. This mode offers the **PDF export** ([src/reporting/pdf.py](src/reporting/pdf.py)) at two depths — `brief` (~2 pages: verdict, money, timeline, actions) and `full` (adds per-dimension evidence with thresholds, ruled-out explanations, product attribution, and a provenance appendix carrying every monthly series).

  The report is **tables only**. Charts were built, reviewed and deliberately dropped: the monthly-series appendix carries the same numbers in quotable form, and the export has no rendering dependency to fail on the demo machine.
- **Answer Key validation** — runs the agent across the reference accounts and scores each verdict against the Answer Key, reporting accuracy split by expected outcome (FLAG / NO FLAG / DEFER). The split matters: one overall number can't distinguish a discriminating agent from one that flags everything. A failed run is recorded as a miss, never dropped from the denominator.
- **View offline demo** — pre-computed reports from `demo_cache/`, the on-stage fallback for no network / no key / rate limit. Built with scripted verdicts, labelled as such in the UI, and **never** silently substituted for a failed live run; the user picks that mode explicitly.
