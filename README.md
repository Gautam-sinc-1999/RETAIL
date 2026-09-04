# Revenue Leakage Investigator

Quessathon "Retail: Where Did the Revenue Go?" challenge. An AI agent that investigates a B2B account's transaction history to find where and why revenue is leaking — even when the account still looks healthy at the top line — and flags when the evidence isn't enough to say for sure.

Built against the official dataset, `Quessathon_Revenue_Leakage_Dataset.xlsx`: 18 Meridian Supply Co accounts, 551 orders, 3,289 transaction lines, Sep 2024 – Aug 2026. Only ~6 of those 18 accounts are genuine structural leaks, so the discrimination — not the alarm — is the product.

See [plan.md](plan.md) for the 7-stage pipeline design, full build log, and current status.

## Setup

```bash
python -m venv venv
venv/Scripts/pip install -r requirements-dev.txt   # venv/bin/pip on POSIX
cp .env.example .env   # fill in GROQ_API_KEY (or ANTHROPIC_API_KEY once available)

venv/Scripts/python scripts/prepare_dataset.py     # workbook -> data/meridian/*.csv
```

## Run

```bash
venv/Scripts/streamlit run app.py
```

Three modes in the sidebar:

- **Analyze a CSV** — upload any transaction file, pick an account, run the live pipeline, and download a PDF report (executive brief or full dossier).
- **Answer Key validation** — run the agent across the reference accounts and score every verdict against the workbook's Answer Key tab. Needs live model access.
- **View offline demo** — pre-computed reports from `demo_cache/`, for when there's no network or key.

## Score the agent against the Answer Key

```bash
venv/Scripts/python scripts/validate_answer_key.py                    # all 18 accounts
venv/Scripts/python scripts/validate_answer_key.py --accounts ACC-101 ACC-109
venv/Scripts/python scripts/validate_answer_key.py --out scorecard.json
```

Accuracy is reported overall and split by expected outcome (FLAG / NO FLAG / DEFER) — a single number can't tell a discriminating agent from one that flags everything. The Answer Key is used only to score finished reports; it is never shown to the agent.

## Test

```bash
venv/Scripts/python -m pytest tests/ -v
```

The suite runs with a scripted stand-in for the model, so it verifies the deterministic stages and the scorecard logic without an API key. It does not measure model judgement — that's what `validate_answer_key.py` is for.
