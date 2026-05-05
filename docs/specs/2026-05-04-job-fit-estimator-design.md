# Job Fit & Salary Estimator — Design Spec

**Status:** Approved 2026-05-04
**Owner:** Petr
**Context:** Czech AI-engineering interview case study deliverable
**Time budget:** 6–8 hours of focused build

---

## 1. Goal

Build a Python pipeline that takes a CV (PDF or DOCX), produces:

- **Seniority score** 0–100, composed of weighted subscores
- **Salary estimate** as a range with confidence label
- **LLM explanation**: strengths, gaps, and concrete actions to grow salary by +30%

The deliverable is judged on *thinking, pipeline design, and handling ambiguity* — not on building "the smartest AI."

## 2. Locked Scope Decisions

| Decision | Value |
|---|---|
| Tier | Mid-tier polished build (~6–8 hrs) |
| Locale | Bilingual (Czech + English CVs, CZK + EUR/USD/GBP outputs) |
| Role coverage | Truly generic (any CV, any occupation) — confidence labels surface uncertainty |
| Salary anchor methodology | ISCO-08 occupational codes + official statistical sources |
| Scoring approach | Hybrid: deterministic for hard signals, LLM-bounded for soft signals |
| LLM | Anthropic Claude Sonnet via `MODEL_ID`, default in `.env.example`; `LLM_PROVIDER` env to swap |
| Interfaces | Streamlit UI core; minimal CLI core; polished CLI + FastAPI optional stretch |

**Headline architecture:** *Official wage anchor + transparent seniority model + bounded LLM explanation.*

## 3. Pipeline

```
PDF/DOCX
   │
   ▼  extract.py      pypdf / python-docx → raw text
   ▼  redact.py       regex strip email/phone/URL/birthdate; sha256 of original
   ▼  parse.py        Claude tool-use → CVJson {roles, skills, edu, langs, certs}
   ▼  classify.py     cached ESCO/ISCO index top-k → Claude pick → confidence + rollup (4→3→2 digit)
   ▼  score.py        Hybrid rubric → ScoreCard {5 subscores + total + band}
   ▼  salary.py       Anchored interpolation → SalaryRange {point, low, high, currency, confidence}
   ▼  recommend.py    +30% branching → GrowthPlan {target_salary, branch, actions}
   ▼
ResultJson → Streamlit / FastAPI / CLI output
```

Every step is a pure function with Pydantic-typed input/output. The orchestrator in `pipeline.py` composes them and applies disk-level caching keyed by `(content_hash, step_name, prompt_version)`.

### 3.1 Extract

`pypdf` for PDF, `python-docx` for DOCX. Returns raw text. No layout reconstruction in v1; if extraction quality is poor, that flows downstream as low parse confidence.

### 3.2 Redact (PII pre-pass)

Regex pre-pass strips PII *before* any LLM call. Patterns: email, phone (international + CZ formats), URLs (incl. LinkedIn), birth dates, ID numbers (rodné číslo).

**Rationale:** GDPR Art. 5 data minimisation. The system does not need names/contacts/addresses to estimate seniority and salary. Stating this explicitly in README is a defensibility win for the interview.

Original file gets a `sha256` hash for traceability and cache keying.

### 3.3 Parse (CV → structured JSON)

Configured Anthropic Claude model (via `MODEL_ID`) with tool-use. Tool schema returns `CVJson`:

```python
class Role(BaseModel):
    title: str
    company: Optional[str]
    start_date: Optional[date]      # may be None for undated entries
    end_date: Optional[date]        # None means current
    description: Optional[str]
    is_current: bool
    raw_dates: Optional[str]        # original date string for debugging
    isco_code: Optional[str] = None       # populated by classify step (not parse)
    isco_confidence: float = 0.0          # populated by classify step

class CVJson(BaseModel):
    roles: list[Role]
    skills: list[str]
    education: list[Education]
    languages: list[Language]
    certifications: list[Certification]
    detected_language: Literal["cs", "en", "other"]
    parse_confidence: float                  # 0.0–1.0 self-reported
    extraction_warnings: list[str] = []      # e.g. "low text length", "no dates found", "PDF extraction likely failed"
```

Prompt-cached system block contains the full extraction rubric and examples. Per-CV input is just the redacted text.

### 3.4 Classify (CV → ISCO)

**Process:**
1. Extract anchor role title + recent role descriptions from `CVJson` (or use user-supplied `target_role`).
2. Query local ESCO occupation index (built from ESCO API, cached) → top-k candidates.
3. Claude picks best match given full CV evidence + top-k candidates.
4. Returns `ISCOClassification`:

```python
class ISCOClassification(BaseModel):
    isco_code: str              # 4, 3, or 2 digit depending on confidence
    isco_level: Literal[2, 3, 4]
    role_label: str             # human-readable
    confidence: float
    top1_margin: float          # gap between top-1 and top-2 candidates
    alternatives: list[str]     # other plausible ISCO codes considered
    rollup_reason: Optional[str] # e.g. "low confidence; rolled up from 4-digit to 3-digit"
```

**Confidence-gated rollup:**

```python
if confidence >= 0.80 and top1_margin >= 0.15:
    isco_level = 4
elif confidence >= 0.60:
    isco_level = 3
else:
    isco_level = 2
```

This prevents fake precision when the CV is ambiguous. Salary distributions widen accordingly.

### 3.5 Anchor role selection (multi-role / career-changer)

```python
def role_duration_months(role: Role, today: date = None) -> int:
    if not role.start_date:
        return 0
    today = today or date.today()
    return months_between(role.start_date, role.end_date or today)

def choose_anchor_role(roles, target_role=None, today=None):
    today = today or date.today()
    if target_role:
        return classify_to_isco(target_role)
    current = [r for r in roles if r.is_current]
    if current:
        return max(current, key=lambda r: role_duration_months(r, today))
    substantial = [r for r in roles if role_duration_months(r, today) >= 12]
    return max(substantial or roles, key=lambda r: r.end_date or today)
```

Career-changer (e.g. nurse 5y → dev 3y) anchors to dev (recent, ≥12 months), not nurse.

## 4. Scoring Rubric

### 4.1 Score blend (weights sum to 1.0)

| Subscore | Weight | Computed by |
|---|---|---|
| `relevant_experience` | 0.25 | Deterministic: interval-union YoE × ISCO similarity |
| `skills_match` | 0.25 | Hybrid: Claude extracts skills, rule-based scoring (breadth + depth + ISCO match) |
| `impact_scope` | 0.20 | LLM-bounded: numeric outcomes, scope of work, business impact |
| `leadership_ownership_growth` | 0.20 | LLM-bounded: 5 sub-dims × 0–20 |
| `education` | 0.10 | Deterministic ordinal |

`total = Σ weight_i × subscore_i`, mapped to band:

| Band | Total |
|---|---|
| Junior | < 40 |
| Mid | 40–60 |
| Senior | 60–80 |
| Lead/Principal | 80–95 |
| Exec | 95+ |

**ScoreCard model:**

```python
class ScoreCard(BaseModel):
    relevant_experience: int
    skills_match: int
    impact_scope: int
    leadership_ownership_growth: int
    education: int
    total: int
    band: Literal["Junior", "Mid", "Senior", "Lead/Principal", "Exec"]
    confidence: Literal["high", "medium", "low"]
    confidence_reasons: list[str]   # e.g. ["missing dates in 2 roles", "ISCO rolled up to 3-digit", "low PDF text quality"]
```

Score confidence is computed independently of salary confidence — they share inputs but answer different questions ("how reliable is the score" vs. "how reliable is the salary anchor").

### 4.2 `relevant_experience` (deterministic)

```python
def total_yoe(roles, today=None):
    today = today or date.today()
    intervals = [(r.start_date, r.end_date or today) for r in roles if r.start_date]
    merged = merge_intervals(sorted(intervals))
    return sum(months_between(s, e) for s, e in merged) / 12

def relevant_yoe(roles, anchor_isco, today=None):
    """FTE-equivalent capped: overlapping roles can't sum to >1.0 per month."""
    today = today or date.today()
    month_weights = {}
    for r in roles:
        if not r.start_date:
            continue
        weight = isco_similarity(r.isco_code, anchor_isco) * max(r.isco_confidence, 0.5)
        for month in iter_months(r.start_date, r.end_date or today):
            month_weights[month] = min(1.0, month_weights.get(month, 0) + weight)
    return sum(month_weights.values()) / 12

def isco_similarity(role_isco, anchor_isco):
    if not role_isco or not anchor_isco: return 0.25
    if role_isco == anchor_isco: return 1.00
    if role_isco[:3] == anchor_isco[:3]: return 0.75
    if role_isco[:2] == anchor_isco[:2]: return 0.50
    if role_isco[:1] == anchor_isco[:1]: return 0.25
    return 0.10
```

`total_yoe` uses interval union (handles parallel jobs).
`relevant_yoe` is **FTE-equivalent capped** — overlapping relevant roles can't accumulate >1.0 contribution per month, so freelance overlap doesn't inflate experience.

`relevant_yoe` → 0–100 via piecewise mapping (e.g. 0y=0, 2y=30, 5y=55, 10y=80, 15y=95, 20+y=100).

### 4.3 `skills_match` (hybrid)

Claude extracts skill list. Rule-based score from three signals:
- **Breadth**: count of distinct skills, normalized
- **Depth**: presence of senior/specialized markers (architecture, mentorship, complex tooling) — extracted by Claude as a 0–20 sub-signal
- **Match**: fraction of CV skills that overlap with ISCO occupation's expected skill set, *if the local ESCO occupation→skill mapping is available*

**Two-tier implementation (graceful degradation):**

```python
def skills_match_score(extracted_skills, isco_code):
    breadth = score_breadth(extracted_skills)             # 0-40
    depth = score_depth(extracted_skills)                 # 0-30
    expected = esco_skills_for(isco_code)                 # may return None
    if expected:
        match = score_overlap(extracted_skills, expected) # 0-30
        return breadth + depth + match, "high"
    else:
        # ESCO skill mapping unavailable → fallback heuristic
        match_proxy = score_keyword_match(extracted_skills, isco_keywords(isco_code))
        return breadth + depth + match_proxy * 0.7, "medium"
```

If ESCO skill mapping is missing or stale, the score uses keyword-group fallback and lowers `skills_match` confidence — feeding into the overall ScoreCard `confidence` label. This means a v1 build can ship without the full ESCO skill index and still produce defensible scores.

### 4.4 `impact_scope` (LLM-bounded)

Claude scores 0–100 against this rubric (passed in prompt):
- 0–20: vague responsibilities, no measurable outcomes
- 21–40: concrete deliverables, no numbers
- 41–60: some measurable results (counts, percentages)
- 61–80: business-level impact (revenue, cost, reliability, scale)
- 81–100: cross-org / multi-million-scale impact

Output must include evidence quotes from the CV. Schema enforces upper bound.

### 4.5 `leadership_ownership_growth` (LLM-bounded, 5 sub-dimensions)

5 dimensions × 0–20 = 0–100:

1. **Ownership** (0–20): task executor → owns features → owns outcomes/budgets
2. **Leadership** (0–20): no signal → mentored/coordinated → led people/strategy/hiring
3. **Learning trajectory** (0–20): stagnant → visible progression → repeated upskilling
4. **Impact clarity** (0–20): vague → concrete deliverables → measurable business results
5. **Stability/execution** (0–20): unexplained hops → normal → sustained delivery + promotions

Each sub-score requires evidence quotes. Sum capped at 100.

### 4.6 `education` (deterministic, role-sensitive)

Education contribution is **role-sensitive** — a PhD should not auto-inflate scoring for roles where formal credentials are not central (e.g. freelance creative, sales, photography), and should weigh heavily where they are (medicine, law, academia, regulated engineering).

**Base score (highest formal education):**
- No listed formal education: 30
- High school / vocational: 45
- Bachelor's: 65
- Master's: 80
- PhD: 90

**Relevance adjustment (vs. anchor ISCO):**
- +10 if directly relevant to anchor occupation, or legally required for the role
- −20 if unrelated to anchor occupation and not generally required
- 0 otherwise

Capped at 100.

**No institution-prestige scoring in v1.** A "top-tier institutions allowlist" introduces regional bias, maintenance overhead, and looks like prestige scoring to reviewers. We score degree level and relevance only.

## 5. Salary Math (the load-bearing logic)

### 5.1 Score → percentile (anchored interpolation)

```python
ANCHORS = [(20, 10), (35, 25), (55, 50), (75, 75), (90, 90)]

def score_to_percentile(score):
    score = max(0, min(100, score))
    if score <= 20:
        return 5 + (score / 20) * 5            # 0→P5, 20→P10
    for (s0, p0), (s1, p1) in zip(ANCHORS, ANCHORS[1:]):
        if s0 <= score <= s1:
            t = (score - s0) / (s1 - s0)
            return p0 + t * (p1 - p0)
    return 90 + ((score - 90) / 10) * 5        # 90→P90, 100→P95 (capped, no extrapolation)
```

**Why anchored, not linear:** A 75/100 candidate is *not* P75 of their ISCO cohort — the cohort already includes P0–P100 at varying scores. Anchored interpolation maps named bands (junior/mid/senior/lead) to their realistic salary percentiles.

### 5.2 Percentile → salary (linear interpolation through ISPV deciles)

```python
def percentile_to_salary(p, d1, q1, median, q3, d9):
    p = max(10, min(90, p))   # never extrapolate past observed deciles
    points = [(10, d1), (25, q1), (50, median), (75, q3), (90, d9)]
    for (p0, v0), (p1, v1) in zip(points, points[1:]):
        if p0 <= p <= p1:
            t = (p - p0) / (p1 - p0)
            return v0 + t * (v1 - v0)
```

Output:

```python
class SalaryRange(BaseModel):
    point: int                   # salary at score-derived percentile
    low: int                     # at (percentile - width)
    high: int                    # at (percentile + width)
    percentile: float            # the score-derived percentile (the point estimate's position)
    percentile_low: float
    percentile_high: float
    currency: Literal["CZK", "EUR", "USD", "GBP"]
    period: Literal["month", "year"]
    confidence: Literal["high", "medium", "low"]
    confidence_reasons: list[str]
    isco_code: str
    isco_level: int
    data_source: str
    data_year: int
```

**Confidence widens the salary band — confidence labels have mathematical effect, not just a UI chip:**

```python
RANGE_WIDTH_BY_CONFIDENCE = {
    "high": 10,    # ±10 percentile points
    "medium": 15,
    "low": 25,
}

width = RANGE_WIDTH_BY_CONFIDENCE[confidence]
percentile_low = max(10, percentile - width)
percentile_high = min(90, percentile + width)
low  = percentile_to_salary(percentile_low,  ...)
high = percentile_to_salary(percentile_high, ...)
```

**Confidence labels:**
- **high**: CZ + ISCO-4 match in ISPV; complete CV with dates
- **medium**: EU + Eurostat SES match, or CZ with rolled-up ISCO-3
- **low**: ISCO rolled to 2-digit, non-EU/non-CZ/non-US/non-UK fallback, or major extraction warnings (missing dates, low text quality)

`confidence_reasons` makes the label auditable, e.g. `["EU country, SES 2022 (quadrennial)", "ISCO rolled up to 3-digit"]`.

## 6. +30% Recommendation (branching)

```python
def growth_plan(current_salary, current_score, isco_distribution):
    target = current_salary * 1.30
    target_p = salary_to_percentile(target, *isco_distribution.deciles())

    # Branch 1: target above observed top decile → must change role family
    if target > isco_distribution.d9:
        return {
            "branch": "role_family_change",
            "message": (
                "+30% target exceeds the top decile of your current ISCO group. "
                "Skill development inside this role is unlikely to deliver +30%. "
                "Consider role-family change, geography, industry, or compensation model."
            ),
            "target_salary": target,
            "alternative_iscos": suggest_higher_paying_iscos(current_isco, transferable_skills),
            "actions": llm_actions(current_cv, alternative_iscos),
        }

    # Branch 2: target at P85+ → technically reachable but realistic recommendation
    # has to combine skill-up with market signals (move company, change industry,
    # geography, comp model) — not just "take a course"
    if target_p >= 85:
        return {
            "branch": "stretch_within_role_or_market_change",
            "message": (
                "+30% target lands in the top 15% of your current ISCO distribution. "
                "Reaching it usually requires a combination of demonstrated impact + "
                "moving company/industry/geography, not skill-up alone."
            ),
            "target_salary": target,
            "target_percentile": target_p,
            "actions": llm_actions(current_cv, mode="stretch"),
        }

    # Branch 3: standard skill-up within role
    required_score = percentile_to_required_score(target_p)
    delta = required_score - current_score
    subscore_deltas = allocate_delta_by_capacity(
        current_subscores, delta, weights,
        skip={"relevant_experience"}    # can't fast-track YoE
    )
    return {
        "branch": "skill_up_within_role",
        "target_salary": target,
        "target_percentile": target_p,
        "required_score": required_score,
        "score_delta": delta,
        "subscore_deltas": subscore_deltas,
        "actions": llm_actions(current_cv, subscore_deltas),
    }
```

LLM is given the deltas + CV evidence and produces 3–5 concrete actions. Constrained: must reference CV evidence, cannot invent credentials, must fit a 6–12 month horizon.

## 7. Data Layer

| Country | Source | Endpoint / File | Format | Refresh |
|---|---|---|---|---|
| CZ | MPSV ISPV (open data) | `data.mpsv.cz/od/soubory/ispv-zamestnani/ispv-zamestnani.json` | JSON | Semi-annual |
| CZ (fallback) | ISPV XLSX (M8r) | `ispv.cz/cz/vysledky-setreni/aktualni.aspx` | XLSX | Semi-annual |
| EU | Eurostat SES 2022 | `earn_ses_main` via `eurostat` PyPI pkg | API | Quadrennial |
| US | BLS OEWS + SOC↔ISCO crosswalk | `bls.gov/oes/` + `bls.gov/soc/isco_soc_crosswalk.xls` | XLS/CSV | Annual |
| UK | ONS SOC 2020 earnings | `ons.gov.uk/employmentandlabourmarket/peopleinwork` | CSV | Annual |
| else | Eurostat fallback + low-confidence label | — | — | — |

**ISPV public dataset is CZ-ISCO × wage/pay sphere only** — not a full region × education × age cube. Region/education adjustments are *not* in v1. README explicitly states this.

Data is fetched once via `scripts/fetch_*.py` and cached as parquet in `data/`. Re-fetch is manual.

ESCO occupation index is built once via `scripts/build_esco_index.py` from the ESCO API and cached locally (sqlite + simple TF-IDF or embedding lookup). The script also fetches the occupation → required-skills mapping used by the `skills_match` subscore in §4.3.

## 8. Interfaces

### 8.1 CLI

```bash
uv run job-fit analyze samples/cvs/mid_dev.pdf \
    --target-role "Senior Data Engineer" \
    --country CZ \
    --json
```

Outputs `ResultJson` to stdout. `--verbose` adds per-step timing and token usage.

### 8.2 FastAPI (optional stretch)

FastAPI is a stretch interface, not core. The core deliverable is Streamlit + CLI. If time permits:

- `POST /analyze` — multipart file upload + optional `target_role`, `country` form fields → `ResultJson`
- `GET /healthz` — basic check

Auth out of scope for v1. If running short on time, this section is skipped without affecting the demo or the marking story — the same pipeline runs through Streamlit and CLI.

### 8.3 Streamlit

Single-page upload → 4 panels:
1. **Score breakdown** — horizontal bar chart with each subscore + total + band label
2. **Salary range** — point estimate with low/high band, currency, confidence chip, data source citation
3. **Strengths & gaps** — LLM explanation grouped into strengths / gaps with CV-quoted evidence
4. **+30% growth plan** — target salary, branch ("skill up" or "change role family"), 3–5 concrete actions as a checklist

Debug expander shows pipeline timings + token usage + cache hits.

## 9. Validation

No labeled ground truth dataset exists. Validation strategy: **adversarial synthetic CV pack** + rank-order assertions.

### 9.1 Synthetic CV pack (10 CVs in `samples/cvs/`)

| File | Profile | Expected band | Key assertion |
|---|---|---|---|
| `junior_dev_1y.pdf` | 1y dev, strong skills, low impact | Junior (25–40) | < mid_dev |
| `mid_dev_4y.pdf` | 4y dev, normal progression | Mid (45–60) | between junior and senior |
| `senior_dev_8y.pdf` | 8y dev, ownership, architecture | Senior (65–80) | > mid_dev |
| `qa_automation_5y.pdf` | 5y QA, strong relevant skills | Mid (50–65) | skills_match high, leadership low |
| `nurse_10y.pdf` | 10y nurse, non-IT senior | Mid–Senior (50–70) | anchors to ISCO 2221 |
| `nurse_to_dev_5y_2y.pdf` | Career changer | Mid (40–55) | relevant_yoe < total_yoe; salary anchor uses developer ISCO code, not historical nurse ISCO code |
| `exec_head_dept.pdf` | Head of dept, P&L | Lead/Principal (80–95) | leadership_ownership ≥ 80 |
| `freelancer_overlap.pdf` | Multiple parallel freelance roles | varies | total_yoe via interval union (no double-counting) |
| `buzzword_no_evidence.pdf` | Verbose, no concrete results | Low (20–40) | impact_scope < 30 |
| `missing_dates.pdf` | Senior content, undated roles | varies | confidence flag = low; salary range wider |

### 9.2 Test types

```python
# Rank assertions
assert score("senior_dev_8y") > score("mid_dev_4y") > score("junior_dev_1y")

# Career-changer: anchor + relevant YoE
result = analyze("nurse_to_dev_5y_2y", target_role="developer")
assert result.anchor.isco_code.startswith("25")  # IT
assert result.relevant_yoe < result.total_yoe

# Buzzword penalty
assert score("buzzword_no_evidence") < score("mid_dev_4y")

# Confidence gating
assert analyze("missing_dates").salary.confidence in ("medium", "low")

# Sensitivity: PII removal stable
assert abs(score(cv) - score(redact_extra(cv))) < 5
```

### 9.3 Per-step unit tests

- `test_yoe.py` — interval union, parallel/overlapping roles, ongoing roles, undated entries
- `test_score_mapping.py` — anchor interpolation correctness; edge cases at 0, 20, 100
- `test_salary_math.py` — percentile↔salary inverse; +30% branching when target > D9
- `test_classify.py` — confidence-gated rollup (4→3→2 digit)
- `test_redact.py` — regex catches expected PII patterns; doesn't false-positive on dates of employment

## 10. Caching, Logging, Config

**Caching:**
- Disk cache `cache/{sha256}.json` keyed by `(content_hash, step_name, prompt_version)`
- Anthropic prompt caching on rubric/instruction blocks (1h TTL)
- Cache invalidated when prompt version bumps

**Logging:**
- `loguru` with structured fields (step, duration_ms, input_hash, tokens_in, tokens_out, cache_hit)
- Streamlit debug expander surfaces per-run trace
- Errors logged with full context, not silently swallowed

**Config:**
- `.env` for API keys (`ANTHROPIC_API_KEY`, optional `OPENAI_API_KEY`)
- `LLM_PROVIDER=anthropic|openai` default `anthropic`
- `MODEL_ID` overridable; default `claude-sonnet-4-6`
- `DATA_DIR` overridable; default `./data`

## 11. Project Structure

```
case-studies/job-fit-estimator/
├── README.md
├── pyproject.toml                # uv-managed, Python 3.11+
├── .env.example
├── data/                         # cached source data (gitignored except for placeholders)
│   ├── ispv/
│   ├── eurostat/
│   ├── bls/
│   └── esco/
├── scripts/
│   ├── fetch_ispv.py
│   ├── fetch_eurostat.py
│   ├── fetch_bls.py
│   └── build_esco_index.py
├── src/job_fit/
│   ├── __init__.py
│   ├── models.py                 # Pydantic: CVJson, Role, ScoreCard, SalaryRange, GrowthPlan, ResultJson
│   ├── extract.py
│   ├── redact.py
│   ├── parse.py
│   ├── classify.py
│   ├── score.py
│   ├── salary.py
│   ├── recommend.py
│   ├── llm.py                    # Anthropic + provider swap
│   ├── cache.py
│   ├── pipeline.py
│   ├── api.py                    # FastAPI
│   └── cli.py
├── app/
│   └── streamlit_app.py
├── samples/
│   └── cvs/                      # 10 synthetic CVs
└── tests/
    ├── test_yoe.py
    ├── test_score_mapping.py
    ├── test_salary_math.py
    ├── test_classify.py
    ├── test_redact.py
    └── test_e2e_synthetic.py
```

## 12. README Structure

0. **Implementation honesty (top of README)** — the framing paragraph reviewers see first:

   > This is an interview-grade prototype, not a production compensation engine. The core implementation focuses on Czech ISPV salary anchoring, transparent scoring, and bounded LLM explanations. International salary sources, richer ESCO skill matching, and FastAPI are implemented only where time permits and otherwise documented as extension points.

   This prevents reviewers from judging unfinished stretch work as failed core work.

1. **What it does** — 3 lines
2. **Run it** — `uv sync && uv run streamlit run app/streamlit_app.py` (and CLI alternative)
3. **Pipeline diagram** — the ASCII flow from §3
4. **Data approach** — sources table, freshness disclosure, confidence labels
5. **Methodology** — score blend, anchored interpolation, +30% branching with worked examples
6. **Limitations** (honest disclosures):
   - No full ISPV multidim cube (CZ-ISCO × sphere only)
   - Eurostat SES is 2022 (quadrennial); v1 may ship CZ-only and document EU/US/UK as extensions
   - "Personality" renamed to leadership/ownership/growth signals
   - No labeled ground truth → validation is rank-order on synthetic pack
   - PII redaction is regex-based (production would use NER)
   - No institution-prestige scoring (deliberate, to avoid bias)
7. **Implementation status** — explicit table of what's "implemented" vs. "designed fallback":
   - Core (implemented): extract → redact → parse → classify (CZ-ISCO) → score → CZ salary → recommend → Streamlit + CLI
   - Stretch (designed, may be partial): EU/US/UK salary sources, full ESCO skill mapping, FastAPI endpoint, all 10 synthetic PDFs (vs. starter 5)
8. **Validation** — synthetic CV pack + assertions
9. **Future work** — not built but obvious extensions (region/edu adjustment when data permits, NER-based redaction, real eval harness with expert panel)

## 13. Implementation Tiers & Out of Scope

### 13.1 Must-have (v1 core, time-boxed at ~5–6 hrs)

1. PDF/DOCX extraction
2. PII redaction
3. Claude parse → CVJson
4. ISCO classification with cached small ESCO subset (CZ-relevant occupations)
5. Cached CZ ISPV sample salary data
6. Deterministic score + salary math (the load-bearing logic of §5)
7. Streamlit UI
8. Minimal CLI wrapper: `uv run job-fit analyze <file> --json`
9. 5 synthetic CVs covering the core profiles (junior_dev, mid_dev, senior_dev, nurse_to_dev, buzzword_no_evidence)
10. Tests for salary math + YoE algorithm

### 13.2 Stretch (only if time permits)

1. Polished CLI binary entry point with `--verbose`, rich output, and `pyproject.toml` console-scripts packaging
2. FastAPI endpoint
3. Eurostat / BLS / ONS salary sources for non-CZ countries
4. Full ESCO occupation→skill mapping (skills_match upgrades from "medium" to "high" confidence)
5. All 10 synthetic CVs (vs. starter 5)
6. Caching layer beyond the LLM prompt cache
7. Comprehensive test suite beyond the math tests

### 13.3 Out of Scope (explicit non-goals)

- Multi-CV batch processing
- Authenticated API
- Database persistence (results are returned, not stored)
- Region/education/age salary adjustments (data not publicly available as a clean cube)
- PPP currency conversion
- ATS-style keyword matching against a specific job posting
- Bias / fairness audit (mentioned in README as future work)
- Production-grade NER for redaction
- Full ISCO-08 4-digit catalog memorization in LLM (we use ESCO retrieval instead)
- Institution-prestige scoring (deliberately excluded — bias risk + maintenance overhead)

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| MPSV ISPV JSON schema changes | Schema validation on fetch; pin to specific snapshot in `data/` |
| ESCO API rate limits or downtime | Local cached index built once; pipeline doesn't hit API at runtime |
| Claude returns malformed JSON | Tool-use with strict Pydantic validation; retry once with error-context prompt |
| LLM hallucinates numbers | Final score and salary are computed deterministically from extracted features; LLM only produces evidence + soft subscores within bounded schema |
| Synthetic CV pack overfits | Rank assertions are loose (band ranges, not exact scores); +1 real anonymized CV during dev as a sanity sample |
| Time overrun | Strict cut at 8 hrs; stretch items in §13.2 are dropped first (FastAPI, full ESCO skill index, non-CZ salary sources); README clearly marks what shipped vs. designed |

## 15. Success Criteria

- At least the 5 core synthetic CVs (§13.1.9) produce sensible bands and rank-order correctly
- If time permits, all 10 synthetic CVs from §9.1 are included and pass rank assertions
- README explains the methodology clearly enough that a reviewer can audit the math
- Pipeline runs E2E on a fresh checkout in under 1 command (`uv sync` then `uv run streamlit run app/streamlit_app.py`)
- All limitations and data caveats stated honestly in README and Streamlit UI
- Tests pass; type checks pass; basic lint passes

## 16. Demo Path

For interview reviewers — the exact path through the system the demo will show:

**Demo input:** `samples/cvs/mid_dev.pdf` with `country = CZ`, no `target_role` (system picks anchor automatically).

**Expected flow:**

1. **Upload** — drop PDF into Streamlit; status banner shows extraction OK + redaction count (e.g. "redacted 2 emails, 1 phone, 3 URLs").
2. **Parse** — extracted role history table renders, `parse_confidence` chip shown, any `extraction_warnings` surfaced.
3. **Classify** — ISCO classification card shows top pick with code, label, confidence, `top1_margin`, alternatives, and `rollup_reason` if rolled up.
4. **Score** — horizontal bar chart with 5 subscores + total + band; ScoreCard `confidence` chip + `confidence_reasons` list.
5. **Salary** — point estimate with low/high band, currency, confidence chip, percentile range, data source citation (e.g. "MPSV ISPV, 1. pololetí 2025"), `confidence_reasons`.
6. **+30% growth plan** — target salary, branch (`skill_up_within_role` / `stretch_within_role_or_market_change` / `role_family_change`), 3–5 concrete actions as a checklist.
7. **Debug expander** — per-step timings, token usage, cache hits, full ResultJson dump.

This makes the demo predictable for reviewers and ensures every must-have section of the spec has a visible UI surface.
