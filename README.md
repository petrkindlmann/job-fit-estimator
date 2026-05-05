# Job Fit & Salary Estimator

> This is an interview-grade prototype, not a production compensation engine.
> The core implementation focuses on Czech ISPV salary anchoring, transparent scoring,
> and bounded LLM explanations. International salary sources, richer ESCO skill matching,
> and FastAPI are implemented only where time permits and otherwise documented as extension points.

**Live artifacts:**
- 🚀 Run it live (upload your own CV): <https://job-fit-estimator.streamlit.app>
- 📊 Sample results on 5 synthetic CVs: <https://job-fit.n8calls.com/results.html>
- 📐 Design explainer (every step, formula, and decision): <https://job-fit.n8calls.com/explainer.html>
- Landing page: <https://job-fit.n8calls.com>

## What it does

Takes a CV (PDF or DOCX), produces:
- A **0–100 seniority score** with audit trail
- A **salary range** (CZK/month) with confidence label, anchored on Czech ISPV statistics
- An **LLM-generated +30% growth plan**: strengths, gaps, 3–5 concrete actions

## Run it

```bash
# 1. Install deps
uv sync --extra dev

# 2. Set up environment
cp .env.example .env
# edit .env to add your ANTHROPIC_API_KEY

# 3. Fetch data (one-time)
uv run python scripts/fetch_ispv.py
uv run python scripts/build_esco_index.py

# 4a. Run the Streamlit UI
uv run streamlit run app/streamlit_app.py

# 4b. ...or use the CLI
uv run job-fit analyze samples/cvs/mid_dev_4y.docx --json
```

### Deploy your own copy to Streamlit Community Cloud

The repo is Streamlit-Cloud-ready (`requirements.txt`, `.streamlit/config.toml`,
sys-path bridging in `app/streamlit_app.py`). One-time setup:

1. Visit <https://share.streamlit.io> and sign in with GitHub.
2. Click **New app**.
3. Repo: `petrkindlmann/job-fit-estimator` · Branch: `main` · Main file path: `app/streamlit_app.py`.
4. **App name** (= subdomain): `job-fit-estimator` (or pick another — first-come-first-served).
5. **Advanced settings → Secrets**: paste the keys from `.streamlit/secrets.toml.example`.
6. Click **Deploy**. The app boots in ~2 minutes; URL: `https://<app-name>.streamlit.app`.

If you pick a different name, update the link in `public/index.html` and `README.md`.

## Pipeline

```
PDF/DOCX
   │
   ▼  extract.py        pypdf / python-docx → raw text
   ▼  redact.py         regex strip email/phone/URL/birthdate; sha256 of original
   ▼  parse.py          Claude tool-use → CVJson
   ▼  classify.py       cached ESCO/ISCO index top-k → Claude pick → confidence-gated rollup
   ▼  score.py          Hybrid rubric → ScoreCard {5 subscores + total + band}
   ▼  salary.py         Anchored interpolation → SalaryRange
   ▼  recommend.py      +30% branching → GrowthPlan
```

Every step is a pure function with Pydantic-typed I/O.

## Data approach

| Country | Source | Notes |
|---|---|---|
| CZ | [MPSV ISPV open data](https://data.mpsv.cz/web/data/ispv-zamestnani) | CZ-ISCO × wage/pay sphere; semi-annual |
| EU/US/UK | Eurostat SES / BLS OEWS / ONS | **Stretch** — not in v1 core |

ISPV gives D1, Q1, median, Q3, D9 per ISCO. We anchor against these observed deciles
and never extrapolate beyond the observed range.

## Methodology

**Scoring rubric (weights sum to 1.0):**

| Subscore | Weight | Computed by |
|---|---|---|
| `relevant_experience` | 0.25 | Deterministic: interval-union YoE × ISCO similarity |
| `skills_match` | 0.25 | Hybrid: keyword-group overlap with ISCO expected skills |
| `impact_scope` | 0.20 | LLM-bounded: numeric outcomes, scope of work |
| `leadership_ownership_growth` | 0.20 | LLM-bounded, 5 sub-dims × 0–20 |
| `education` | 0.10 | Deterministic, role-sensitive |

**Score → salary mapping (anchored interpolation):**

A 75/100 candidate is NOT P75 of their ISCO cohort. We map named seniority bands to
realistic salary percentiles:

```
Score   →   Percentile
20            P10  (entry)
35            P25  (junior)
55            P50  (mid)
75            P75  (senior)
90            P90  (lead/principal)
```

Then percentile → salary via linear interpolation through observed ISPV deciles.

**+30% growth plan branches:**

1. `skill_up_within_role` — target inside current ISCO distribution; LLM gives 3–5 concrete actions tied to subscore deltas.
2. `stretch_within_role_or_market_change` — target ≥ P85; combines skill-up with company/industry/geography moves.
3. `role_family_change` — target above current ISCO's D9; recommendation is to pivot occupation, not skill up.

## Implementation status

| Component | Status |
|---|---|
| PDF/DOCX extraction | ✅ Implemented |
| PII redaction | ✅ Implemented |
| Claude parse → CVJson | ✅ Implemented |
| ISCO classification (CZ-relevant subset) | ✅ Implemented |
| CZ ISPV salary anchor | ✅ Implemented |
| Deterministic score + salary math | ✅ Implemented |
| Streamlit UI | ✅ Implemented |
| Minimal CLI | ✅ Implemented |
| 5 synthetic CVs | ✅ Implemented |
| EU/US/UK salary sources | ⏸️ Designed (stretch) |
| Full ESCO occupation→skill mapping | ⏸️ Designed (stretch) |
| FastAPI endpoint | ⏸️ Designed (stretch) |
| 5 additional adversarial synthetic CVs | ⏸️ Designed (stretch) |

## Limitations

- **No full ISPV multidim cube** — public dataset is CZ-ISCO × wage sphere only. Region/education/age adjustments are designed but not in v1.
- **Eurostat SES is 2022 (quadrennial)** — international anchors will lag.
- **"Personality" renamed to leadership/ownership/growth signals** — CV text cannot defensibly infer personality; CV-observable signals are the honest proxy.
- **No labeled ground truth** — validation is rank-order on a synthetic pack (§9 of spec).
- **Regex PII redaction** — production would use NER.
- **No institution-prestige scoring** — deliberately excluded to avoid bias.

## Validation

`tests/test_e2e_synthetic.py` runs the full pipeline on the 5 core synthetic CVs and
asserts rank ordering (junior < mid < senior, buzzword < mid, career-changer's relevant
YoE < total YoE, etc.). Run locally with `ANTHROPIC_API_KEY` set.

Pure-function tests (`test_yoe.py`, `test_score_mapping.py`, `test_salary_math.py`,
`test_recommend.py`, etc.) run without an API key.

```bash
uv run pytest tests/ -v --ignore=tests/test_e2e_synthetic.py    # unit
ANTHROPIC_API_KEY=... uv run pytest tests/test_e2e_synthetic.py -v   # e2e
```

### HTML test + coverage reports

```bash
uv run pytest tests/ --ignore=tests/test_e2e_synthetic.py \
    --html=docs/reports/test-report.html --self-contained-html \
    --cov=src/job_fit --cov-report=html:docs/reports/coverage --cov-report=term
open docs/reports/test-report.html        # browse test results
open docs/reports/coverage/index.html     # browse coverage by file/line
```

Latest run: 71 unit tests pass, 86% line coverage. 6 E2E tests pass when API key is set.

## Design explainer

A standalone walkthrough of every step, every formula, and every decision lives at:

- HTML: [`docs/explainer.html`](docs/explainer.html) — open in a browser
- PDF:  [`docs/explainer.pdf`](docs/explainer.pdf) — generated from the HTML via Playwright

To regenerate the PDF after editing the HTML:

```bash
node scripts/render_pdf.mjs
```

## Future work

- Region / education / age salary adjustments when public CZ data permits
- NER-based PII redaction
- Real eval harness with expert panel
- ESCO occupation→skill mapping for richer skills_match
- Bias / fairness audit
- CZ-trained ISCO classifier vs. zero-shot LLM benchmark

## Spec

Full design rationale: `docs/specs/2026-05-04-job-fit-estimator-design.md`
