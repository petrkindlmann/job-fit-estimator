# Job Fit & Salary Estimator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interview-grade Python pipeline that takes a CV (PDF/DOCX), produces a 0–100 seniority score, a salary range with confidence label, and an LLM-generated +30% growth plan.

**Architecture:** Pure-function pipeline (extract → redact → parse → classify → score → salary → recommend) with Pydantic-typed I/O at each step. Deterministic core math (anchored interpolation through ISPV deciles), LLM bounded to evidence extraction and soft subscores within strict schemas. Czech-first salary anchor (MPSV ISPV open data) with EU/US/UK as stretch.

**Tech Stack:** Python 3.11+, `uv` for env/deps, Pydantic v2, Anthropic SDK (tool-use), `pypdf`, `python-docx`, `streamlit`, `loguru`, `pytest`, `ruff`, `mypy`. FastAPI for stretch.

**Spec:** `docs/specs/2026-05-04-job-fit-estimator-design.md` (read this first; tasks reference its sections).

**Time budget:** 5–6 hrs core (Tasks 1–24), 2 hrs stretch (Tasks 25–30).

**Project root:** All commands run from `/Users/petr/projects/N8-upwork/case-studies/job-fit-estimator/`.

---

## Phase 0 — Scaffolding

### Task 1: Initialize project & dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/job_fit/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Init git repo**

```bash
cd /Users/petr/projects/N8-upwork/case-studies/job-fit-estimator
git init
git branch -m main
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "job-fit"
version = "0.1.0"
description = "CV → seniority score + salary estimate + growth plan"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "pydantic>=2.7",
    "pypdf>=4.2",
    "python-docx>=1.1",
    "streamlit>=1.36",
    "loguru>=0.7",
    "python-dotenv>=1.0",
    "httpx>=0.27",
    "pandas>=2.2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov", "ruff>=0.5", "mypy>=1.10"]
api = ["fastapi>=0.111", "python-multipart>=0.0.9", "uvicorn>=0.30"]

[project.scripts]
job-fit = "job_fit.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/job_fit"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = false
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic
MODEL_ID=claude-sonnet-4-6
DATA_DIR=./data
LOG_LEVEL=INFO
```

- [ ] **Step 4: Write `.gitignore`**

```
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
data/cache/
data/ispv/raw/
data/eurostat/raw/
data/bls/raw/
data/esco/raw/
*.egg-info/
dist/
build/
.coverage
htmlcov/
```

- [ ] **Step 5: Create empty package files**

```bash
mkdir -p src/job_fit tests app scripts samples/cvs data/ispv data/esco data/cache docs/specs docs/plans
touch src/job_fit/__init__.py tests/__init__.py
```

- [ ] **Step 6: Sync deps and verify**

```bash
uv sync --extra dev
uv run python -c "import anthropic, pydantic, streamlit, pypdf, docx; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example .gitignore src/ tests/ docs/specs/
git commit -m "chore: scaffold project (pyproject, dirs, deps)"
```

---

### Task 2: Pydantic models (the contract layer)

**Files:**
- Create: `src/job_fit/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_models.py
from datetime import date
from job_fit.models import Role, CVJson, ISCOClassification, ScoreCard, SalaryRange, GrowthPlan, ResultJson


def test_role_minimum():
    r = Role(title="Dev", is_current=True)
    assert r.title == "Dev"
    assert r.start_date is None
    assert r.isco_code is None
    assert r.isco_confidence == 0.0


def test_cvjson_with_warnings():
    cv = CVJson(
        roles=[Role(title="Dev", is_current=True)],
        skills=["python"],
        education=[],
        languages=[],
        certifications=[],
        detected_language="en",
        parse_confidence=0.8,
        extraction_warnings=["no dates found"],
    )
    assert cv.parse_confidence == 0.8
    assert "no dates found" in cv.extraction_warnings


def test_scorecard_band_values():
    sc = ScoreCard(
        relevant_experience=70, skills_match=70, impact_scope=60,
        leadership_ownership_growth=60, education=70,
        total=66, band="Senior", confidence="high", confidence_reasons=[],
    )
    assert sc.band == "Senior"


def test_salary_range_percentile_fields():
    sr = SalaryRange(
        point=80000, low=70000, high=90000,
        percentile=60.0, percentile_low=50.0, percentile_high=70.0,
        currency="CZK", period="month", confidence="high",
        confidence_reasons=["CZ ISCO-4 match"],
        isco_code="2511", isco_level=4,
        data_source="MPSV ISPV", data_year=2025,
    )
    assert sr.point == 80000
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/test_models.py -v
```

Expected: ImportError / module not found.

- [ ] **Step 3: Implement `src/job_fit/models.py`**

```python
from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Role(BaseModel):
    title: str
    company: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    description: Optional[str] = None
    is_current: bool = False
    raw_dates: Optional[str] = None
    isco_code: Optional[str] = None
    isco_confidence: float = 0.0


class Education(BaseModel):
    degree: str           # "Bachelor's", "Master's", "PhD", "High school", "Vocational", "None"
    field: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class Language(BaseModel):
    code: str             # ISO 639-1 best-effort
    proficiency: Optional[str] = None   # "native", "C2", "B2", etc.


class Certification(BaseModel):
    name: str
    issuer: Optional[str] = None
    year: Optional[int] = None


class CVJson(BaseModel):
    roles: list[Role]
    skills: list[str]
    education: list[Education]
    languages: list[Language]
    certifications: list[Certification]
    detected_language: Literal["cs", "en", "other"]
    parse_confidence: float = Field(ge=0.0, le=1.0)
    extraction_warnings: list[str] = []


class ISCOClassification(BaseModel):
    isco_code: str
    isco_level: Literal[2, 3, 4]
    role_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    top1_margin: float = 0.0
    alternatives: list[str] = []
    rollup_reason: Optional[str] = None


class ScoreCard(BaseModel):
    relevant_experience: int = Field(ge=0, le=100)
    skills_match: int = Field(ge=0, le=100)
    impact_scope: int = Field(ge=0, le=100)
    leadership_ownership_growth: int = Field(ge=0, le=100)
    education: int = Field(ge=0, le=100)
    total: int = Field(ge=0, le=100)
    band: Literal["Junior", "Mid", "Senior", "Lead/Principal", "Exec"]
    confidence: Literal["high", "medium", "low"]
    confidence_reasons: list[str] = []
    evidence: dict[str, list[str]] = {}     # subscore_name -> [evidence quotes]


class SalaryRange(BaseModel):
    point: int
    low: int
    high: int
    percentile: float
    percentile_low: float
    percentile_high: float
    currency: Literal["CZK", "EUR", "USD", "GBP"]
    period: Literal["month", "year"]
    confidence: Literal["high", "medium", "low"]
    confidence_reasons: list[str] = []
    isco_code: str
    isco_level: int
    data_source: str
    data_year: int


class GrowthPlan(BaseModel):
    branch: Literal["skill_up_within_role", "stretch_within_role_or_market_change", "role_family_change"]
    target_salary: int
    target_percentile: Optional[float] = None
    required_score: Optional[int] = None
    score_delta: Optional[int] = None
    subscore_deltas: dict[str, float] = {}
    message: str
    actions: list[str]


class ResultJson(BaseModel):
    cv: CVJson
    classification: ISCOClassification
    score: ScoreCard
    salary: SalaryRange
    growth_plan: GrowthPlan
    pipeline_meta: dict = {}    # timings, token usage, cache hits
```

- [ ] **Step 4: Run tests, verify pass**

```bash
uv run pytest tests/test_models.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/models.py tests/test_models.py
git commit -m "feat(models): pydantic contracts for pipeline I/O"
```

---

## Phase 1 — Pure-function core (TDD-friendly, no external dependencies)

### Task 3: PDF/DOCX text extraction

**Files:**
- Create: `src/job_fit/extract.py`
- Test: `tests/test_extract.py`
- Test fixtures: `tests/fixtures/sample.pdf`, `tests/fixtures/sample.docx`

- [ ] **Step 1: Create test fixtures**

```bash
mkdir -p tests/fixtures
# Use uv run python to create a tiny PDF and DOCX with known text:
uv run python -c "
from docx import Document
d = Document()
d.add_paragraph('John Doe')
d.add_paragraph('Senior Developer')
d.add_paragraph('5 years Python experience')
d.save('tests/fixtures/sample.docx')
"
# For PDF, write a simple one via reportlab or use a pre-made tiny PDF.
# Quick path: use pypdf to convert from a markdown-ish text via fpdf2:
uv add --dev fpdf2
uv run python -c "
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font('Helvetica', size=12)
for line in ['John Doe', 'Senior Developer', '5 years Python experience']:
    pdf.cell(0, 10, line, ln=True)
pdf.output('tests/fixtures/sample.pdf')
"
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_extract.py
from pathlib import Path
from job_fit.extract import extract_text

FIX = Path(__file__).parent / "fixtures"


def test_extract_pdf():
    text = extract_text(FIX / "sample.pdf")
    assert "Senior Developer" in text
    assert "Python" in text


def test_extract_docx():
    text = extract_text(FIX / "sample.docx")
    assert "Senior Developer" in text
    assert "Python" in text


def test_extract_unsupported_extension(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    import pytest
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text(p)
```

- [ ] **Step 3: Run test, verify fail**

```bash
uv run pytest tests/test_extract.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `src/job_fit/extract.py`**

```python
from pathlib import Path
from pypdf import PdfReader
from docx import Document


def extract_text(path: Path | str) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(p)
    if suffix == ".docx":
        return _extract_docx(p)
    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()


def _extract_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs).strip()
```

- [ ] **Step 5: Run tests, verify pass**

```bash
uv run pytest tests/test_extract.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/job_fit/extract.py tests/test_extract.py tests/fixtures/
git commit -m "feat(extract): pdf/docx text extraction"
```

---

### Task 4: PII redaction

**Files:**
- Create: `src/job_fit/redact.py`
- Test: `tests/test_redact.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_redact.py
from job_fit.redact import redact_pii, hash_bytes


def test_redact_email():
    text = "Contact: john.doe@example.com for info"
    out = redact_pii(text)
    assert "john.doe@example.com" not in out
    assert "[REDACTED_EMAIL]" in out


def test_redact_phone_intl():
    text = "Call +420 777 123 456 anytime"
    out = redact_pii(text)
    assert "777 123 456" not in out
    assert "[REDACTED_PHONE]" in out


def test_redact_url_and_linkedin():
    text = "Profile https://linkedin.com/in/johndoe and site www.example.com"
    out = redact_pii(text)
    assert "linkedin.com/in/johndoe" not in out
    assert "[REDACTED_LINKEDIN]" in out or "[REDACTED_URL]" in out


def test_redact_birth_date_cs():
    text = "Datum narození: 12.03.1990"
    out = redact_pii(text)
    assert "12.03.1990" not in out
    assert "[REDACTED_BIRTH_DATE]" in out


def test_redact_does_not_strip_employment_dates():
    text = "Worked at ACME from 03/2018 to 06/2022 as engineer"
    out = redact_pii(text)
    assert "03/2018" in out and "06/2022" in out


def test_hash_bytes_stable():
    h1 = hash_bytes(b"hello")
    h2 = hash_bytes(b"hello")
    assert h1 == h2 and len(h1) == 64
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_redact.py -v
```

- [ ] **Step 3: Implement `src/job_fit/redact.py`**

```python
import hashlib
import re

PII_PATTERNS: list[tuple[str, str]] = [
    # ORDER MATTERS: linkedin before generic url; birth_date before phone (date numbers can look phone-ish).
    ("BIRTH_DATE", r"(?:datum narození|date of birth|born|narozen[aá]?)\s*[:\-]?\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"),
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("LINKEDIN", r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s)]+"),
    ("URL", r"https?://\S+|www\.\S+"),
    # CZ phone: optional +420, then groups of digits separated by space/-/dot
    ("PHONE", r"(?:(?:\+|00)\d{1,3}[\s.-]?)?(?:\d{3}[\s.-]?\d{3}[\s.-]?\d{3}|\(?\d{2,4}\)?[\s.-]?\d{3}[\s.-]?\d{3,4})"),
    # CZ rodné číslo (birth ID) — 6 digits / 3-4 digits
    ("CZ_ID", r"\b\d{6}\s?/\s?\d{3,4}\b"),
]


def redact_pii(text: str) -> str:
    out = text
    for label, pattern in PII_PATTERNS:
        out = re.sub(pattern, f"[REDACTED_{label}]", out, flags=re.IGNORECASE)
    return out


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_redact.py -v
```

Expected: 6 tests pass. If `test_redact_does_not_strip_employment_dates` fails, the phone regex is too greedy — tighten it.

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/redact.py tests/test_redact.py
git commit -m "feat(redact): regex PII pre-pass + sha256 hash"
```

---

### Task 5: Score → percentile (anchored interpolation)

**Files:**
- Create: `src/job_fit/salary_math.py`
- Test: `tests/test_score_mapping.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_score_mapping.py
import pytest
from job_fit.salary_math import score_to_percentile, ANCHORS


def test_score_anchor_points():
    assert score_to_percentile(20) == pytest.approx(10, abs=0.01)
    assert score_to_percentile(35) == pytest.approx(25, abs=0.01)
    assert score_to_percentile(55) == pytest.approx(50, abs=0.01)
    assert score_to_percentile(75) == pytest.approx(75, abs=0.01)
    assert score_to_percentile(90) == pytest.approx(90, abs=0.01)


def test_score_interp_midpoint():
    # Score 65 lies between (55,50) and (75,75) → P62.5
    assert score_to_percentile(65) == pytest.approx(62.5, abs=0.5)


def test_score_below_floor_clipped():
    p = score_to_percentile(0)
    assert 0 <= p <= 10


def test_score_above_ceiling_capped():
    p = score_to_percentile(100)
    assert 90 <= p <= 95


def test_score_negative_clamped():
    assert score_to_percentile(-5) == score_to_percentile(0)


def test_score_overflow_clamped():
    assert score_to_percentile(150) == score_to_percentile(100)
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_score_mapping.py -v
```

- [ ] **Step 3: Implement `src/job_fit/salary_math.py`**

```python
from typing import Optional

ANCHORS: list[tuple[float, float]] = [
    (20, 10),
    (35, 25),
    (55, 50),
    (75, 75),
    (90, 90),
]


def score_to_percentile(score: float) -> float:
    """Map seniority score (0-100) to salary distribution percentile (5-95).

    Anchored, not linear: a 75/100 candidate is NOT P75 of the cohort. Anchor points
    map named bands (junior/mid/senior/lead) to realistic percentiles.
    """
    score = max(0.0, min(100.0, float(score)))
    if score <= 20:
        return 5 + (score / 20) * 5  # 0→P5, 20→P10
    if score >= 90:
        return 90 + ((score - 90) / 10) * 5  # 90→P90, 100→P95
    for (s0, p0), (s1, p1) in zip(ANCHORS, ANCHORS[1:]):
        if s0 <= score <= s1:
            t = (score - s0) / (s1 - s0)
            return p0 + t * (p1 - p0)
    raise RuntimeError("unreachable")  # pragma: no cover


def percentile_to_required_score(target_p: float) -> int:
    """Inverse of score_to_percentile. Used for +30% recommendation math."""
    target_p = max(5.0, min(95.0, float(target_p)))
    inverse = [(p, s) for s, p in ANCHORS]
    # Add the soft endpoints so we can hit P5–P10 and P90–P95 ranges.
    inverse = [(5.0, 0.0)] + inverse + [(95.0, 100.0)]
    for (p0, s0), (p1, s1) in zip(inverse, inverse[1:]):
        if p0 <= target_p <= p1:
            t = (target_p - p0) / (p1 - p0)
            return int(round(s0 + t * (s1 - s0)))
    raise RuntimeError("unreachable")  # pragma: no cover
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_score_mapping.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/salary_math.py tests/test_score_mapping.py
git commit -m "feat(salary): score→percentile anchored interpolation"
```

---

### Task 6: Percentile → salary + range widening by confidence

**Files:**
- Modify: `src/job_fit/salary_math.py` (append)
- Test: `tests/test_salary_math.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_salary_math.py
import pytest
from job_fit.salary_math import (
    percentile_to_salary, salary_to_percentile,
    salary_range_from_percentile, RANGE_WIDTH_BY_CONFIDENCE,
)

DECILES = dict(d1=40000, q1=55000, median=75000, q3=105000, d9=150000)


def test_percentile_to_salary_at_anchors():
    assert percentile_to_salary(10, **DECILES) == pytest.approx(40000)
    assert percentile_to_salary(50, **DECILES) == pytest.approx(75000)
    assert percentile_to_salary(90, **DECILES) == pytest.approx(150000)


def test_percentile_to_salary_p625():
    # P62.5 lies between P50 (75k) and P75 (105k) → 75k + 0.5*(30k) = 90k
    assert percentile_to_salary(62.5, **DECILES) == pytest.approx(90000)


def test_percentile_clamped_below_p10():
    assert percentile_to_salary(5, **DECILES) == pytest.approx(40000)


def test_percentile_clamped_above_p90():
    assert percentile_to_salary(99, **DECILES) == pytest.approx(150000)


def test_salary_to_percentile_inverse():
    # Round-trip: 90k → P62.5 → 90k
    p = salary_to_percentile(90000, **DECILES)
    assert p == pytest.approx(62.5, abs=0.5)


def test_range_widens_with_lower_confidence():
    high = salary_range_from_percentile(60, "high", **DECILES)
    low = salary_range_from_percentile(60, "low", **DECILES)
    assert (high.high - high.low) < (low.high - low.low)


def test_range_width_table():
    assert RANGE_WIDTH_BY_CONFIDENCE["high"] == 10
    assert RANGE_WIDTH_BY_CONFIDENCE["medium"] == 15
    assert RANGE_WIDTH_BY_CONFIDENCE["low"] == 25
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_salary_math.py -v
```

- [ ] **Step 3: Append to `src/job_fit/salary_math.py`**

```python
from dataclasses import dataclass

RANGE_WIDTH_BY_CONFIDENCE: dict[str, int] = {
    "high": 10,
    "medium": 15,
    "low": 25,
}


@dataclass
class SalaryBand:
    point: int
    low: int
    high: int
    percentile: float
    percentile_low: float
    percentile_high: float


def percentile_to_salary(p: float, *, d1: float, q1: float, median: float, q3: float, d9: float) -> float:
    """Linear interpolation through observed ISPV deciles. Never extrapolates beyond P10/P90."""
    p = max(10.0, min(90.0, float(p)))
    points = [(10.0, d1), (25.0, q1), (50.0, median), (75.0, q3), (90.0, d9)]
    for (p0, v0), (p1, v1) in zip(points, points[1:]):
        if p0 <= p <= p1:
            t = (p - p0) / (p1 - p0)
            return v0 + t * (v1 - v0)
    raise RuntimeError("unreachable")  # pragma: no cover


def salary_to_percentile(salary: float, *, d1: float, q1: float, median: float, q3: float, d9: float) -> float:
    """Inverse of percentile_to_salary. Used for +30% target percentile lookup."""
    points = [(10.0, d1), (25.0, q1), (50.0, median), (75.0, q3), (90.0, d9)]
    if salary <= d1:
        return 10.0
    if salary >= d9:
        # Soft pseudo-percentile above D9 (capped) — used only to detect "above top decile" branch.
        return 90.0 + min(10.0, (salary - d9) / d9 * 50.0)
    for (p0, v0), (p1, v1) in zip(points, points[1:]):
        if v0 <= salary <= v1:
            t = (salary - v0) / (v1 - v0)
            return p0 + t * (p1 - p0)
    raise RuntimeError("unreachable")  # pragma: no cover


def salary_range_from_percentile(
    percentile: float,
    confidence: str,
    *,
    d1: float, q1: float, median: float, q3: float, d9: float,
) -> SalaryBand:
    width = RANGE_WIDTH_BY_CONFIDENCE[confidence]
    p_low = max(10.0, percentile - width)
    p_high = min(90.0, percentile + width)
    point = int(round(percentile_to_salary(percentile, d1=d1, q1=q1, median=median, q3=q3, d9=d9)))
    low = int(round(percentile_to_salary(p_low, d1=d1, q1=q1, median=median, q3=q3, d9=d9)))
    high = int(round(percentile_to_salary(p_high, d1=d1, q1=q1, median=median, q3=q3, d9=d9)))
    return SalaryBand(point=point, low=low, high=high,
                      percentile=percentile, percentile_low=p_low, percentile_high=p_high)
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_salary_math.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/salary_math.py tests/test_salary_math.py
git commit -m "feat(salary): percentile→salary + confidence-widened range"
```

---

### Task 7: Years-of-experience (interval union + FTE-capped relevant YoE)

**Files:**
- Create: `src/job_fit/yoe.py`
- Test: `tests/test_yoe.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_yoe.py
from datetime import date
import pytest
from job_fit.models import Role
from job_fit.yoe import total_yoe, relevant_yoe, isco_similarity, role_duration_months


TODAY = date(2026, 5, 4)


def role(title, start, end=None, isco=None, conf=1.0, current=False):
    return Role(
        title=title, start_date=start, end_date=end,
        is_current=current, isco_code=isco, isco_confidence=conf,
    )


def test_total_yoe_simple():
    rs = [role("Dev", date(2020, 1, 1), date(2024, 1, 1))]
    assert total_yoe(rs, today=TODAY) == pytest.approx(4.0, abs=0.1)


def test_total_yoe_interval_union_no_double_count():
    # Two parallel roles overlapping completely → counts as one period
    rs = [
        role("FT", date(2020, 1, 1), date(2024, 1, 1)),
        role("Side", date(2021, 1, 1), date(2023, 1, 1)),
    ]
    assert total_yoe(rs, today=TODAY) == pytest.approx(4.0, abs=0.1)


def test_total_yoe_ongoing_role():
    rs = [role("Cur", date(2024, 5, 4), end=None, current=True)]
    assert total_yoe(rs, today=TODAY) == pytest.approx(2.0, abs=0.1)


def test_total_yoe_ignores_undated():
    rs = [role("Past", None), role("Dev", date(2022, 5, 4), date(2024, 5, 4))]
    assert total_yoe(rs, today=TODAY) == pytest.approx(2.0, abs=0.1)


def test_isco_similarity_levels():
    assert isco_similarity("2511", "2511") == 1.00
    assert isco_similarity("2511", "2512") == 0.75   # same 3-digit
    assert isco_similarity("2511", "2521") == 0.50   # same 2-digit
    assert isco_similarity("2511", "3511") == 0.25   # same 1-digit
    assert isco_similarity("2511", "5111") == 0.10   # different
    assert isco_similarity(None, "2511") == 0.25     # unknown


def test_relevant_yoe_career_changer_lower_than_total():
    # 5y nurse (anchor=dev) + 3y dev
    rs = [
        role("Nurse", date(2017, 1, 1), date(2022, 1, 1), isco="2221"),
        role("Dev", date(2022, 1, 1), date(2025, 1, 1), isco="2511"),
    ]
    t = total_yoe(rs, today=TODAY)
    r = relevant_yoe(rs, anchor_isco="2511", today=TODAY)
    assert r < t
    # Dev gets 3y * 1.0 + nurse 5y * 0.10 = ~3.5y
    assert r == pytest.approx(3.5, abs=0.5)


def test_relevant_yoe_fte_capped():
    # Two parallel dev roles (full overlap) shouldn't double the relevant YoE
    rs = [
        role("Dev1", date(2020, 1, 1), date(2024, 1, 1), isco="2511"),
        role("Dev2", date(2020, 1, 1), date(2024, 1, 1), isco="2511"),
    ]
    r = relevant_yoe(rs, anchor_isco="2511", today=TODAY)
    assert r == pytest.approx(4.0, abs=0.1)


def test_role_duration_months():
    r = role("X", date(2020, 1, 1), date(2022, 1, 1))
    assert role_duration_months(r, TODAY) == 24

    r2 = role("Y", None)
    assert role_duration_months(r2, TODAY) == 0
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_yoe.py -v
```

- [ ] **Step 3: Implement `src/job_fit/yoe.py`**

```python
from datetime import date
from typing import Iterator
from job_fit.models import Role


def months_between(a: date, b: date) -> int:
    return max(0, (b.year - a.year) * 12 + (b.month - a.month))


def role_duration_months(role: Role, today: date | None = None) -> int:
    if not role.start_date:
        return 0
    today = today or date.today()
    end = role.end_date or today
    return months_between(role.start_date, end)


def merge_intervals(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[list[date]] = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def total_yoe(roles: list[Role], today: date | None = None) -> float:
    today = today or date.today()
    intervals = [(r.start_date, r.end_date or today) for r in roles if r.start_date]
    merged = merge_intervals(intervals)
    months = sum(months_between(s, e) for s, e in merged)
    return months / 12.0


def isco_similarity(role_isco: str | None, anchor_isco: str | None) -> float:
    if not role_isco or not anchor_isco:
        return 0.25
    if role_isco == anchor_isco:
        return 1.00
    if role_isco[:3] == anchor_isco[:3]:
        return 0.75
    if role_isco[:2] == anchor_isco[:2]:
        return 0.50
    if role_isco[:1] == anchor_isco[:1]:
        return 0.25
    return 0.10


def _iter_months(start: date, end: date) -> Iterator[tuple[int, int]]:
    """Yield (year, month) tuples from start (inclusive) to end (exclusive)."""
    y, m = start.year, start.month
    while (y, m) < (end.year, end.month):
        yield (y, m)
        m += 1
        if m > 12:
            m = 1
            y += 1


def relevant_yoe(roles: list[Role], anchor_isco: str, today: date | None = None) -> float:
    """FTE-equivalent capped: overlapping relevant roles can't sum to >1.0 per month."""
    today = today or date.today()
    month_weights: dict[tuple[int, int], float] = {}
    for r in roles:
        if not r.start_date:
            continue
        weight = isco_similarity(r.isco_code, anchor_isco) * max(r.isco_confidence, 0.5)
        end = r.end_date or today
        for ym in _iter_months(r.start_date, end):
            month_weights[ym] = min(1.0, month_weights.get(ym, 0.0) + weight)
    return sum(month_weights.values()) / 12.0
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_yoe.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/yoe.py tests/test_yoe.py
git commit -m "feat(yoe): interval-union total + FTE-capped relevant YoE"
```

---

### Task 8: Anchor role selection

**Files:**
- Create: `src/job_fit/anchor.py`
- Test: `tests/test_anchor.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_anchor.py
from datetime import date
from job_fit.models import Role
from job_fit.anchor import choose_anchor_role


TODAY = date(2026, 5, 4)


def role(title, start, end=None, isco=None, current=False):
    return Role(title=title, start_date=start, end_date=end,
                is_current=current, isco_code=isco)


def test_anchor_uses_current_role():
    rs = [
        role("Old", date(2018, 1, 1), date(2022, 1, 1), isco="2511"),
        role("Cur", date(2023, 1, 1), end=None, isco="2521", current=True),
    ]
    a = choose_anchor_role(rs, today=TODAY)
    assert a.title == "Cur"


def test_anchor_falls_back_to_substantial_recent():
    rs = [
        role("Short", date(2024, 12, 1), date(2025, 3, 1), isco="2511"),
        role("Long", date(2020, 1, 1), date(2023, 12, 1), isco="2521"),
    ]
    a = choose_anchor_role(rs, today=TODAY)
    assert a.title == "Long"


def test_anchor_returns_most_recent_when_none_substantial():
    rs = [role("A", date(2025, 1, 1), date(2025, 3, 1), isco="2511"),
          role("B", date(2025, 4, 1), date(2025, 6, 1), isco="2521")]
    a = choose_anchor_role(rs, today=TODAY)
    assert a.title == "B"
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_anchor.py -v
```

- [ ] **Step 3: Implement `src/job_fit/anchor.py`**

```python
from datetime import date
from job_fit.models import Role
from job_fit.yoe import role_duration_months


def choose_anchor_role(roles: list[Role], today: date | None = None) -> Role:
    today = today or date.today()
    if not roles:
        raise ValueError("no roles to anchor on")
    current = [r for r in roles if r.is_current]
    if current:
        return max(current, key=lambda r: role_duration_months(r, today))
    substantial = [r for r in roles if role_duration_months(r, today) >= 12]
    pool = substantial or roles
    # Prefer most recent end_date (None ranks highest as "still ongoing")
    return max(pool, key=lambda r: r.end_date or today)
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_anchor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/anchor.py tests/test_anchor.py
git commit -m "feat(anchor): role selection for multi-role/career-changer"
```

---

### Task 9: Education subscore (deterministic, role-sensitive)

**Files:**
- Create: `src/job_fit/score_education.py`
- Test: `tests/test_score_education.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_score_education.py
from job_fit.models import Education
from job_fit.score_education import score_education


def test_no_education():
    assert score_education([], anchor_isco="2511") == 30


def test_phd_unrelated_decreases():
    # PhD with unrelated field, anchor is software dev
    edu = [Education(degree="PhD", field="Philosophy")]
    s = score_education(edu, anchor_isco="2511")  # 90 - 20 = 70
    assert s == 70


def test_phd_related_increases():
    edu = [Education(degree="PhD", field="Computer Science")]
    s = score_education(edu, anchor_isco="2511")  # 90 + 10, capped to 100
    assert s == 100


def test_bachelor_neutral():
    edu = [Education(degree="Bachelor's", field="Marketing")]
    s = score_education(edu, anchor_isco="2511")
    assert s == 65 - 20  # unrelated penalty


def test_picks_highest_degree():
    edu = [
        Education(degree="High school"),
        Education(degree="Master's", field="Computer Science"),
    ]
    s = score_education(edu, anchor_isco="2511")
    assert s == 80 + 10  # master's + relevance bump
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_score_education.py -v
```

- [ ] **Step 3: Implement `src/job_fit/score_education.py`**

```python
from job_fit.models import Education

DEGREE_BASE: dict[str, int] = {
    "none": 30,
    "high school": 45,
    "vocational": 45,
    "bachelor's": 65,
    "bachelor": 65,
    "master's": 80,
    "master": 80,
    "phd": 90,
    "doctorate": 90,
}

# ISCO-08 major groups where formal education is centrally required.
# 2 = Professionals (engineers, doctors, scientists, lawyers, teachers)
# 3 = Technicians (often regulated; e.g. nursing technicians, paralegals)
# 6 = Skilled agricultural (less)
# Below this, formal degrees are usually not central.
EDUCATION_CENTRAL_MAJORS = {"2", "3"}

# Field keywords by ISCO 2-digit (rough relevance map; deliberately conservative)
FIELD_RELEVANCE: dict[str, list[str]] = {
    "25": ["computer", "software", "informatic", "informa", "data", "ai", "machine learning"],
    "21": ["engineering", "physics", "math", "science"],
    "22": ["medicine", "nursing", "health", "pharmacy", "dental"],
    "23": ["education", "teaching", "pedagog"],
    "24": ["finance", "accounting", "business", "marketing", "hr", "law"],
    "26": ["law", "language", "literature", "history", "philosophy"],
}


def _normalize_degree(d: str) -> str:
    return (d or "").strip().lower()


def _base_for(degree: str) -> int:
    norm = _normalize_degree(degree)
    for key, val in DEGREE_BASE.items():
        if key in norm:
            return val
    return DEGREE_BASE["none"]


def _is_relevant(field: str | None, anchor_isco: str) -> bool:
    if not field:
        return False
    field_l = field.lower()
    keywords = FIELD_RELEVANCE.get(anchor_isco[:2], [])
    return any(k in field_l for k in keywords)


def score_education(education: list[Education], anchor_isco: str) -> int:
    if not education:
        return DEGREE_BASE["none"]
    # Pick the highest-ranking degree
    top = max(education, key=lambda e: _base_for(e.degree))
    base = _base_for(top.degree)

    # Relevance adjustment
    if _is_relevant(top.field, anchor_isco):
        base += 10
    elif anchor_isco[:1] not in EDUCATION_CENTRAL_MAJORS:
        # In role families where formal education isn't central, unrelated degrees don't help
        base -= 20

    return max(0, min(100, base))
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_score_education.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/score_education.py tests/test_score_education.py
git commit -m "feat(score): education subscore (role-sensitive)"
```

---

### Task 10: +30% recommendation branching

**Files:**
- Create: `src/job_fit/recommend.py`
- Test: `tests/test_recommend.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_recommend.py
from job_fit.recommend import compute_growth_branch, allocate_subscore_deltas


DECILES = dict(d1=40000, q1=55000, median=75000, q3=105000, d9=150000)


def test_branch_role_family_when_target_above_d9():
    # 130k current * 1.3 = 169k > d9 (150k)
    branch = compute_growth_branch(current_salary=130000, **DECILES)
    assert branch.name == "role_family_change"


def test_branch_stretch_when_target_at_p85():
    # 95k * 1.3 = 123.5k → between q3 and d9 → P~82–86 (depends)
    branch = compute_growth_branch(current_salary=95000, **DECILES)
    # Should be stretch or skill_up; if target_p >= 85 then stretch
    assert branch.name in ("stretch_within_role_or_market_change", "skill_up_within_role")
    if branch.target_percentile >= 85:
        assert branch.name == "stretch_within_role_or_market_change"


def test_branch_skill_up_when_target_in_range():
    # 50k * 1.3 = 65k → P~37
    branch = compute_growth_branch(current_salary=50000, **DECILES)
    assert branch.name == "skill_up_within_role"
    assert branch.target_percentile < 85


def test_allocate_skips_relevant_experience():
    subscores = {
        "relevant_experience": 50, "skills_match": 60,
        "impact_scope": 70, "leadership_ownership_growth": 50, "education": 75,
    }
    weights = {
        "relevant_experience": 0.25, "skills_match": 0.25,
        "impact_scope": 0.20, "leadership_ownership_growth": 0.20, "education": 0.10,
    }
    deltas = allocate_subscore_deltas(subscores, required_total_delta=10, weights=weights,
                                      skip={"relevant_experience"})
    assert deltas["relevant_experience"] == 0
    assert sum(deltas.values()) > 0


def test_allocate_zero_delta_returns_zeros():
    subscores = {"a": 50, "b": 60}
    weights = {"a": 0.5, "b": 0.5}
    deltas = allocate_subscore_deltas(subscores, 0, weights, skip=set())
    assert all(v == 0 for v in deltas.values())
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_recommend.py -v
```

- [ ] **Step 3: Implement `src/job_fit/recommend.py`**

```python
from dataclasses import dataclass, field
from typing import Iterable
from job_fit.salary_math import salary_to_percentile, percentile_to_required_score


@dataclass
class GrowthBranch:
    name: str          # "skill_up_within_role" | "stretch_within_role_or_market_change" | "role_family_change"
    target_salary: int
    target_percentile: float
    message: str
    required_score: int | None = None
    score_delta: int | None = None


def compute_growth_branch(
    current_salary: float,
    *, d1: float, q1: float, median: float, q3: float, d9: float,
    current_score: float | None = None,
    target_multiplier: float = 1.30,
) -> GrowthBranch:
    target = current_salary * target_multiplier
    target_p = salary_to_percentile(target, d1=d1, q1=q1, median=median, q3=q3, d9=d9)

    if target > d9:
        return GrowthBranch(
            name="role_family_change",
            target_salary=int(round(target)),
            target_percentile=target_p,
            message=(
                "+30% target exceeds the top decile of your current ISCO group. "
                "Skill development inside this role is unlikely to deliver +30%. "
                "Consider role-family change, geography, industry, or compensation model."
            ),
        )

    if target_p >= 85:
        return GrowthBranch(
            name="stretch_within_role_or_market_change",
            target_salary=int(round(target)),
            target_percentile=target_p,
            message=(
                "+30% target lands in the top 15% of your current ISCO distribution. "
                "Reaching it usually requires a combination of demonstrated impact + "
                "moving company/industry/geography, not skill-up alone."
            ),
        )

    required = percentile_to_required_score(target_p)
    delta = max(0, required - int(round(current_score or 0)))
    return GrowthBranch(
        name="skill_up_within_role",
        target_salary=int(round(target)),
        target_percentile=target_p,
        required_score=required,
        score_delta=delta,
        message=(
            f"+30% is reachable inside your current ISCO with a score increase of "
            f"~{delta} points (target P{target_p:.0f})."
        ),
    )


def allocate_subscore_deltas(
    subscores: dict[str, float],
    required_total_delta: float,
    weights: dict[str, float],
    skip: Iterable[str],
) -> dict[str, float]:
    """Distribute a required total-score delta across subscores by weighted capacity.
    Skipped subscores (e.g. relevant_experience — can't fast-track YoE) get 0 delta."""
    if required_total_delta <= 0:
        return {k: 0.0 for k in subscores}
    skip_set = set(skip)
    gaps = {k: max(0.0, 100 - v) for k, v in subscores.items() if k not in skip_set}
    weighted_capacity = {k: gaps[k] * weights[k] for k in gaps}
    total_capacity = sum(weighted_capacity.values()) or 1.0
    plan: dict[str, float] = {k: 0.0 for k in subscores}
    for k in gaps:
        share = required_total_delta * (weighted_capacity[k] / total_capacity)
        # Convert score-points-of-total back to subscore-points: divide by weight
        raw = share / weights[k] if weights[k] > 0 else 0.0
        plan[k] = round(min(gaps[k], raw), 1)
    return plan
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_recommend.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/recommend.py tests/test_recommend.py
git commit -m "feat(recommend): +30% branching + subscore delta allocation"
```

---

## Phase 2 — Data layer (CZ-only core)

### Task 11: Fetch & load Czech ISPV salary data

**Files:**
- Create: `scripts/fetch_ispv.py`
- Create: `src/job_fit/data/__init__.py`
- Create: `src/job_fit/data/ispv.py`
- Test: `tests/test_ispv.py`

- [ ] **Step 1: Write fetch script**

```python
# scripts/fetch_ispv.py
"""Fetch MPSV ISPV open data and cache as parquet.

Source: https://data.mpsv.cz/od/soubory/ispv-zamestnani/ispv-zamestnani.json
Schema: CZ-ISCO × wage/pay sphere distributions (D1, Q1, median, Q3, D9, mean, count, period).
"""
from pathlib import Path
import json
import sys
import httpx
import pandas as pd

URL = "https://data.mpsv.cz/od/soubory/ispv-zamestnani/ispv-zamestnani.json"
OUT_RAW = Path("data/ispv/raw/ispv-zamestnani.json")
OUT_PARQUET = Path("data/ispv/ispv.parquet")


def fetch():
    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120) as client:
        r = client.get(URL)
        r.raise_for_status()
        OUT_RAW.write_bytes(r.content)
    payload = json.loads(OUT_RAW.read_text())
    # MPSV uses "polozky" (items) as the array key. Adapt if schema differs.
    rows = payload.get("polozky") or payload.get("items") or payload
    if not isinstance(rows, list):
        raise SystemExit(f"Unexpected ISPV schema: {type(rows)}")
    df = pd.json_normalize(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {len(df)} rows → {OUT_PARQUET}")


if __name__ == "__main__":
    fetch()
```

- [ ] **Step 2: Run fetch script**

```bash
uv run python scripts/fetch_ispv.py
```

Expected: writes `data/ispv/ispv.parquet` with thousands of rows.

If the JSON schema differs from `polozky`, inspect:
```bash
uv run python -c "import json; print(list(json.loads(open('data/ispv/raw/ispv-zamestnani.json').read()).keys())[:10])"
```
and adjust the script accordingly.

- [ ] **Step 3: Write failing test for ISPV lookup**

```python
# tests/test_ispv.py
from job_fit.data.ispv import IspvIndex


def test_index_loads_and_finds_isco():
    idx = IspvIndex.load_default()
    # Pick any ISCO present in the dataset for a smoke test
    assert len(idx.iscos()) > 50

    # Lookup a developer-ish code if present; otherwise use the first ISCO
    target = "2512" if "2512" in idx.iscos() else next(iter(idx.iscos()))
    rec = idx.lookup(target)
    assert rec is not None
    assert rec.median > 0
    assert rec.d1 < rec.median < rec.d9


def test_lookup_with_rollup():
    idx = IspvIndex.load_default()
    iscos = idx.iscos()
    target = next(iter(iscos))
    rec = idx.lookup_with_rollup(target)
    assert rec is not None
    assert rec.isco_level in (2, 3, 4)


def test_lookup_unknown_returns_none():
    idx = IspvIndex.load_default()
    rec = idx.lookup("9999")  # invalid
    assert rec is None
```

- [ ] **Step 4: Run, verify fail**

```bash
uv run pytest tests/test_ispv.py -v
```

- [ ] **Step 5: Implement `src/job_fit/data/__init__.py`** (empty)

```bash
touch src/job_fit/data/__init__.py
```

- [ ] **Step 6: Implement `src/job_fit/data/ispv.py`**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd


@dataclass
class IspvRecord:
    isco_code: str
    isco_level: int    # 2, 3, or 4
    d1: float
    q1: float
    median: float
    q3: float
    d9: float
    mean: float
    period: str        # e.g. "1. pololetí 2025"
    sphere: str        # "MZDOVA" | "PLATOVA"
    count: int


# Field-name mappings; the actual MPSV schema uses these keys.
FIELD_MAP = {
    "isco":   ["czIsco", "cz_isco", "isco_code", "isco"],
    "d1":     ["diferenciaceD1M", "d1", "decile_1"],
    "q1":     ["diferenciaceQ1M", "q1", "quartile_1"],
    "median": ["medianMzda", "medianPlat", "median"],
    "q3":     ["diferenciaceQ3M", "q3", "quartile_3"],
    "d9":     ["diferenciaceD9M", "d9", "decile_9"],
    "mean":   ["mzdaPrumer", "platPrumer", "mean", "average"],
    "period": ["obdobi", "period"],
    "sphere": ["sfera", "sphere"],
    "count":  ["pocetZamestnancu", "employee_count", "count"],
}


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved = {}
    for canon, candidates in FIELD_MAP.items():
        for c in candidates:
            if c in df.columns:
                resolved[canon] = c
                break
    missing = set(FIELD_MAP) - set(resolved)
    if missing:
        # Allow optional fields (sphere, count, period); raise on hard ones
        hard_missing = missing - {"sphere", "count", "period"}
        if hard_missing:
            raise RuntimeError(f"ISPV columns not found: {hard_missing}")
    return resolved


def _normalize_isco(code: str) -> str:
    """Strip 'CzIsco/' prefix if present, keep only digits."""
    s = str(code).split("/")[-1]
    return "".join(ch for ch in s if ch.isdigit())


class IspvIndex:
    DEFAULT_PATH = Path("data/ispv/ispv.parquet")

    def __init__(self, df: pd.DataFrame):
        cols = _resolve_columns(df)
        self.df = df.copy()
        self.df["_isco"] = self.df[cols["isco"]].astype(str).map(_normalize_isco)
        self.df["_isco_level"] = self.df["_isco"].str.len()
        self._cols = cols
        # Prefer the most recent period available; if no period column, keep all rows.
        if "period" in cols:
            most_recent = self.df[cols["period"]].mode()
            if len(most_recent) > 0:
                self.df = self.df[self.df[cols["period"]] == most_recent.iloc[0]]
        self._by_isco = {row["_isco"]: row for _, row in self.df.iterrows()}

    @classmethod
    def load_default(cls) -> "IspvIndex":
        return cls.load(cls.DEFAULT_PATH)

    @classmethod
    def load(cls, path: Path) -> "IspvIndex":
        df = pd.read_parquet(path)
        return cls(df)

    def iscos(self) -> set[str]:
        return set(self._by_isco.keys())

    def _row_to_record(self, row) -> IspvRecord:
        c = self._cols
        return IspvRecord(
            isco_code=row["_isco"],
            isco_level=int(row["_isco_level"]),
            d1=float(row[c["d1"]]),
            q1=float(row[c["q1"]]),
            median=float(row[c["median"]]),
            q3=float(row[c["q3"]]),
            d9=float(row[c["d9"]]),
            mean=float(row[c["mean"]]),
            period=str(row[c["period"]]) if "period" in c else "",
            sphere=str(row[c["sphere"]]) if "sphere" in c else "",
            count=int(row[c["count"]]) if "count" in c else 0,
        )

    def lookup(self, isco: str) -> Optional[IspvRecord]:
        isco_n = _normalize_isco(isco)
        row = self._by_isco.get(isco_n)
        if row is None:
            return None
        return self._row_to_record(row)

    def lookup_with_rollup(self, isco: str) -> Optional[IspvRecord]:
        """Try 4-digit, then 3-digit, then 2-digit prefixes. Aggregate when needed."""
        isco_n = _normalize_isco(isco)
        for level in (4, 3, 2):
            prefix = isco_n[:level]
            exact = self._by_isco.get(prefix)
            if exact is not None:
                return self._row_to_record(exact)
            # Aggregate by averaging records under this prefix
            matches = [r for k, r in self._by_isco.items() if k.startswith(prefix)]
            if matches:
                df_match = pd.DataFrame(matches)
                c = self._cols
                return IspvRecord(
                    isco_code=prefix,
                    isco_level=level,
                    d1=float(df_match[c["d1"]].mean()),
                    q1=float(df_match[c["q1"]].mean()),
                    median=float(df_match[c["median"]].mean()),
                    q3=float(df_match[c["q3"]].mean()),
                    d9=float(df_match[c["d9"]].mean()),
                    mean=float(df_match[c["mean"]].mean()),
                    period=str(df_match[c["period"]].iloc[0]) if "period" in c else "",
                    sphere="AGGREGATE",
                    count=int(df_match[c["count"]].sum()) if "count" in c else 0,
                )
        return None
```

- [ ] **Step 7: Run, verify pass**

```bash
uv run pytest tests/test_ispv.py -v
```

If the field-map doesn't match the live JSON, inspect a few raw rows and add the actual key names to `FIELD_MAP`.

- [ ] **Step 8: Commit**

```bash
git add scripts/fetch_ispv.py src/job_fit/data/ tests/test_ispv.py data/ispv/ispv.parquet
git commit -m "feat(data): MPSV ISPV fetch + indexed lookup with rollup"
```

---

### Task 12: Build small ESCO/ISCO occupation index

**Files:**
- Create: `scripts/build_esco_index.py`
- Create: `src/job_fit/data/esco.py`
- Test: `tests/test_esco.py`

- [ ] **Step 1: Write build script**

```python
# scripts/build_esco_index.py
"""Build a small local index of ISCO-08 occupations + role labels.

For v1 we ship a hand-curated CSV of common occupations spanning the major groups.
Running this script seeds data/esco/occupations.csv if missing. Stretch: pull from
ESCO API at https://ec.europa.eu/esco/api/resource/occupation
"""
from pathlib import Path
import csv

OUT = Path("data/esco/occupations.csv")

# Minimal seed: 4-digit ISCO codes covering common occupations across major groups.
# Format: isco_code,label_en,label_cs,keywords_en,keywords_cs
SEED = [
    ("1120", "Managing director", "Generální ředitel", "ceo, managing director, executive", "ředitel, ceo"),
    ("1330", "ICT services manager", "Manažer IT služeb", "head of engineering, vp engineering, cto", "vedoucí it, manažer it"),
    ("2221", "Nursing professional", "Všeobecná zdravotní sestra", "registered nurse, nurse", "zdravotní sestra, sestra"),
    ("2310", "University and higher education teacher", "Vysokoškolský učitel", "professor, lecturer", "profesor, docent"),
    ("2330", "Secondary education teacher", "Učitel střední školy", "high school teacher", "středoškolský učitel"),
    ("2411", "Accountant", "Účetní", "accountant, controller", "účetní"),
    ("2421", "Management consultant", "Manažerský poradce", "management consultant, strategy consultant", "konzultant, poradce"),
    ("2434", "ICT sales professional", "Obchodník v IT", "sales engineer, account executive, ict sales", "obchodník it, sales it"),
    ("2511", "Systems analyst", "Systémový analytik", "systems analyst, business analyst", "systémový analytik, business analytik"),
    ("2512", "Software developer", "Vývojář software", "software developer, software engineer, backend, full stack, frontend", "programátor, vývojář, software engineer"),
    ("2513", "Web and multimedia developer", "Webový vývojář", "web developer, frontend developer", "webový vývojář, frontend"),
    ("2514", "Applications programmer", "Aplikační programátor", "applications programmer, mobile developer", "aplikační programátor"),
    ("2519", "Software and applications developer NEC", "Ostatní vývojáři SW", "QA engineer, test engineer, automation, sdet", "QA inženýr, tester, automatizace testování"),
    ("2521", "Database designer and administrator", "DBA", "database administrator, dba, data engineer", "dba, datový inženýr"),
    ("2522", "Systems administrator", "Systémový administrátor", "systems administrator, sysadmin, devops, sre", "sysadmin, devops, sre"),
    ("2523", "Computer network professional", "Síťový specialista", "network engineer", "síťový inženýr"),
    ("2529", "Database and network professionals NEC", "Ostatní DB/sítě", "data scientist, ml engineer, ai engineer, machine learning", "data scientist, ml inženýr, AI inženýr"),
    ("2611", "Lawyer", "Právník", "lawyer, attorney", "právník, advokát"),
    ("2632", "Sociologist, anthropologist", "Sociolog", "sociologist, anthropologist", "sociolog"),
    ("2641", "Author and related writer", "Spisovatel", "author, copywriter, content writer", "autor, copywriter"),
    ("2651", "Visual artist", "Výtvarný umělec", "graphic designer, visual artist", "grafik, designer"),
    ("3221", "Nursing associate professional", "Praktická sestra", "nursing assistant, practical nurse", "praktická sestra"),
    ("3322", "Commercial sales representative", "Obchodní zástupce", "sales representative, account manager", "obchodní zástupce, account manager"),
    ("4110", "General office clerk", "Administrativní pracovník", "office administrator, secretary", "administrativní pracovník, sekretářka"),
    ("5120", "Cook", "Kuchař", "cook, chef", "kuchař"),
    ("5223", "Shop sales assistant", "Prodavač", "shop assistant, retail", "prodavač"),
    ("7115", "Carpenter", "Tesař", "carpenter", "tesař"),
    ("8322", "Car driver", "Řidič osobních aut", "driver, taxi driver", "řidič"),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"{OUT} already exists; not overwriting. Delete to regenerate.")
        return
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["isco_code", "label_en", "label_cs", "keywords_en", "keywords_cs"])
        for row in SEED:
            w.writerow(row)
    print(f"Wrote {len(SEED)} occupations → {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run build script**

```bash
uv run python scripts/build_esco_index.py
```

- [ ] **Step 3: Write failing test**

```python
# tests/test_esco.py
from job_fit.data.esco import EscoIndex


def test_index_loads():
    idx = EscoIndex.load_default()
    assert len(idx.candidates) >= 25


def test_search_developer_keywords():
    idx = EscoIndex.load_default()
    top = idx.search("senior python software engineer with 5 years backend", k=5)
    codes = [c.isco_code for c in top]
    assert "2512" in codes  # software developer should rank in top 5


def test_search_nurse_cs():
    idx = EscoIndex.load_default()
    top = idx.search("zdravotní sestra na pohotovosti", k=5)
    codes = [c.isco_code for c in top]
    assert "2221" in codes
```

- [ ] **Step 4: Run, verify fail**

```bash
uv run pytest tests/test_esco.py -v
```

- [ ] **Step 5: Implement `src/job_fit/data/esco.py`**

```python
from dataclasses import dataclass
from pathlib import Path
import csv
import re


@dataclass
class EscoCandidate:
    isco_code: str
    label_en: str
    label_cs: str
    keywords: list[str]
    score: float = 0.0


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-záčďéěíňóřšťúůýž]+", (text or "").lower()))


class EscoIndex:
    DEFAULT_PATH = Path("data/esco/occupations.csv")

    def __init__(self, candidates: list[EscoCandidate]):
        self.candidates = candidates
        self._token_sets = [_tokenize(c.label_en + " " + c.label_cs + " " + " ".join(c.keywords))
                            for c in candidates]

    @classmethod
    def load_default(cls) -> "EscoIndex":
        return cls.load(cls.DEFAULT_PATH)

    @classmethod
    def load(cls, path: Path) -> "EscoIndex":
        cands = []
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kws = [k.strip() for k in (row["keywords_en"] + "," + row["keywords_cs"]).split(",") if k.strip()]
                cands.append(EscoCandidate(
                    isco_code=row["isco_code"],
                    label_en=row["label_en"],
                    label_cs=row["label_cs"],
                    keywords=kws,
                ))
        return cls(cands)

    def search(self, query: str, k: int = 5) -> list[EscoCandidate]:
        q_tokens = _tokenize(query)
        scored: list[EscoCandidate] = []
        for cand, ts in zip(self.candidates, self._token_sets):
            if not ts:
                continue
            overlap = len(q_tokens & ts)
            if overlap == 0:
                continue
            # Jaccard-ish + bonus for keyword exact substring match
            jaccard = overlap / max(1, len(q_tokens | ts))
            substring_bonus = 0.0
            ql = query.lower()
            for kw in cand.keywords:
                if kw.lower() in ql:
                    substring_bonus += 0.1
            scored_cand = EscoCandidate(
                isco_code=cand.isco_code, label_en=cand.label_en, label_cs=cand.label_cs,
                keywords=cand.keywords, score=jaccard + substring_bonus,
            )
            scored.append(scored_cand)
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]

    def by_code(self, isco: str) -> EscoCandidate | None:
        for c in self.candidates:
            if c.isco_code == isco:
                return c
        return None
```

- [ ] **Step 6: Run, verify pass**

```bash
uv run pytest tests/test_esco.py -v
```

- [ ] **Step 7: Commit**

```bash
git add scripts/build_esco_index.py src/job_fit/data/esco.py tests/test_esco.py data/esco/occupations.csv
git commit -m "feat(data): seed ESCO/ISCO occupation index with simple search"
```

---

## Phase 3 — LLM-bounded steps

### Task 13: Anthropic LLM client wrapper

**Files:**
- Create: `src/job_fit/llm.py`
- Test: `tests/test_llm.py` (smoke test only — needs API key, skip if absent)

- [ ] **Step 1: Implement `src/job_fit/llm.py`**

```python
import os
from dataclasses import dataclass
from typing import Any
import anthropic
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


@dataclass
class LlmResponse:
    text: str
    tool_use: dict[str, Any] | None
    usage: dict[str, int]
    cache_hit: bool


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _model() -> str:
    return os.environ.get("MODEL_ID", "claude-sonnet-4-6")


def call_with_tool(
    *,
    system: str,
    user: str,
    tool_name: str,
    tool_schema: dict[str, Any],
    cache_system: bool = True,
    max_tokens: int = 2000,
) -> LlmResponse:
    """Single-tool call. Forces tool_use; returns parsed tool input.

    `cache_system=True` enables Anthropic prompt caching on the system block —
    used heavily because rubrics are reused across CVs.
    """
    sys_block: list[dict[str, Any]] = [{"type": "text", "text": system}]
    if cache_system:
        sys_block[0]["cache_control"] = {"type": "ephemeral"}

    client = _client()
    resp = client.messages.create(
        model=_model(),
        max_tokens=max_tokens,
        system=sys_block,
        tools=[{"name": tool_name, "input_schema": tool_schema, "description": "Single result tool."}],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )

    tool_input: dict[str, Any] | None = None
    text_parts: list[str] = []
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            tool_input = block.input
        elif block.type == "text":
            text_parts.append(block.text)

    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
    }
    cache_hit = usage["cache_read_input_tokens"] > 0
    logger.debug("LLM call: tool={} usage={} cache_hit={}", tool_name, usage, cache_hit)

    return LlmResponse(
        text="\n".join(text_parts),
        tool_use=tool_input,
        usage=usage,
        cache_hit=cache_hit,
    )
```

- [ ] **Step 2: Write smoke test (skipped without API key)**

```python
# tests/test_llm.py
import os
import pytest
from job_fit.llm import call_with_tool


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no API key in env")
def test_smoke_tool_call():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    resp = call_with_tool(
        system="You answer in one word.",
        user="What is the capital of France?",
        tool_name="answer",
        tool_schema=schema,
        max_tokens=200,
    )
    assert resp.tool_use is not None
    assert "paris" in resp.tool_use["answer"].lower()
```

- [ ] **Step 3: Run (will skip if no key, that's fine)**

```bash
uv run pytest tests/test_llm.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/job_fit/llm.py tests/test_llm.py
git commit -m "feat(llm): anthropic client wrapper with tool-use + prompt cache"
```

---

### Task 14: Parse step (text → CVJson via Claude tool-use)

**Files:**
- Create: `src/job_fit/parse.py`
- Test: `tests/test_parse.py` (mocked — no live API in unit tests)

- [ ] **Step 1: Write failing test (with mocked LLM)**

```python
# tests/test_parse.py
from unittest.mock import patch
from job_fit.llm import LlmResponse
from job_fit.parse import parse_cv


MOCK_RESP = LlmResponse(
    text="",
    tool_use={
        "roles": [{"title": "Software Developer", "company": "ACME",
                   "start_date": "2020-01-01", "end_date": "2024-01-01",
                   "is_current": False, "description": "Built things"}],
        "skills": ["python", "sql"],
        "education": [{"degree": "Bachelor's", "field": "Computer Science"}],
        "languages": [{"code": "en", "proficiency": "C1"}],
        "certifications": [],
        "detected_language": "en",
        "parse_confidence": 0.85,
        "extraction_warnings": [],
    },
    usage={"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    cache_hit=False,
)


def test_parse_cv_returns_cvjson():
    with patch("job_fit.parse.call_with_tool", return_value=MOCK_RESP):
        cv = parse_cv("Some CV text...")
    assert len(cv.roles) == 1
    assert cv.roles[0].title == "Software Developer"
    assert "python" in cv.skills
    assert cv.detected_language == "en"


def test_parse_cv_handles_warnings():
    resp = LlmResponse(
        text="",
        tool_use={**MOCK_RESP.tool_use, "extraction_warnings": ["no dates found"], "parse_confidence": 0.4},
        usage=MOCK_RESP.usage, cache_hit=False,
    )
    with patch("job_fit.parse.call_with_tool", return_value=resp):
        cv = parse_cv("Vague CV")
    assert "no dates found" in cv.extraction_warnings
    assert cv.parse_confidence == 0.4
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_parse.py -v
```

- [ ] **Step 3: Implement `src/job_fit/parse.py`**

```python
from job_fit.llm import call_with_tool
from job_fit.models import CVJson


PARSE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": ["string", "null"]},
                    "start_date": {"type": ["string", "null"], "description": "ISO date YYYY-MM-DD or null"},
                    "end_date": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "is_current": {"type": "boolean"},
                    "raw_dates": {"type": ["string", "null"]},
                },
                "required": ["title", "is_current"],
            },
        },
        "skills": {"type": "array", "items": {"type": "string"}},
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "field": {"type": ["string", "null"]},
                    "institution": {"type": ["string", "null"]},
                    "start_date": {"type": ["string", "null"]},
                    "end_date": {"type": ["string", "null"]},
                },
                "required": ["degree"],
            },
        },
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "proficiency": {"type": ["string", "null"]},
                },
                "required": ["code"],
            },
        },
        "certifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "issuer": {"type": ["string", "null"]},
                    "year": {"type": ["integer", "null"]},
                },
                "required": ["name"],
            },
        },
        "detected_language": {"type": "string", "enum": ["cs", "en", "other"]},
        "parse_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "extraction_warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["roles", "skills", "education", "languages", "certifications",
                 "detected_language", "parse_confidence", "extraction_warnings"],
}


SYSTEM_PROMPT = """You are extracting structured CV data into a strict JSON schema.

Rules:
- The CV text has had PII redacted (emails, phones, URLs, birth dates show as [REDACTED_*]). Ignore these tokens.
- Extract every distinct role/job. For ongoing roles, set is_current=true and end_date=null.
- Use ISO YYYY-MM-DD for dates. If only month/year, use the first day of the month.
- If a date is missing or unparseable, set null and add a warning to extraction_warnings.
- Skills: include only technical/professional skills mentioned, deduplicated, lowercased.
- detected_language: "cs" if predominantly Czech, "en" if predominantly English, else "other".
- parse_confidence: 0.0–1.0 reflecting how complete and parseable the CV was.
- extraction_warnings: short tags like "no dates found", "low text length", "PDF extraction likely failed".
- Do NOT invent roles, skills, or dates. If unclear, omit and warn.
- Return all results via the parse_cv tool.
"""


def parse_cv(redacted_text: str) -> CVJson:
    resp = call_with_tool(
        system=SYSTEM_PROMPT,
        user=redacted_text,
        tool_name="parse_cv",
        tool_schema=PARSE_TOOL_SCHEMA,
        max_tokens=4000,
    )
    if resp.tool_use is None:
        raise RuntimeError("LLM did not produce parse_cv tool call")
    return CVJson.model_validate(resp.tool_use)
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_parse.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/parse.py tests/test_parse.py
git commit -m "feat(parse): claude tool-use → CVJson"
```

---

### Task 15: Classify step (CV → ISCO with confidence-gated rollup)

**Files:**
- Create: `src/job_fit/classify.py`
- Test: `tests/test_classify.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_classify.py
from unittest.mock import patch
from job_fit.llm import LlmResponse
from job_fit.models import CVJson, Role
from job_fit.classify import classify_cv, _gate_rollup


def cv(*, role_title="Senior Software Developer"):
    return CVJson(
        roles=[Role(title=role_title, is_current=True)],
        skills=["python"],
        education=[],
        languages=[],
        certifications=[],
        detected_language="en",
        parse_confidence=0.9,
    )


HIGH_CONF_RESP = LlmResponse(
    text="",
    tool_use={"isco_code": "2512", "role_label": "Software developer",
              "confidence": 0.92, "top1_margin": 0.30,
              "alternatives": ["2511", "2519"]},
    usage={"input_tokens": 100, "output_tokens": 30, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    cache_hit=False,
)


def test_classify_high_confidence_keeps_4_digit():
    with patch("job_fit.classify.call_with_tool", return_value=HIGH_CONF_RESP):
        c = classify_cv(cv())
    assert c.isco_code == "2512"
    assert c.isco_level == 4


def test_classify_rolls_up_when_low_confidence():
    low = LlmResponse(
        text="",
        tool_use={"isco_code": "2512", "role_label": "Software developer",
                  "confidence": 0.55, "top1_margin": 0.05,
                  "alternatives": ["2511", "2519", "2434"]},
        usage=HIGH_CONF_RESP.usage, cache_hit=False,
    )
    with patch("job_fit.classify.call_with_tool", return_value=low):
        c = classify_cv(cv())
    assert c.isco_level == 2  # rolled up to 2-digit
    assert c.isco_code == "25"
    assert c.rollup_reason is not None


def test_gate_rollup_thresholds():
    assert _gate_rollup(0.90, 0.20) == 4
    assert _gate_rollup(0.70, 0.10) == 3
    assert _gate_rollup(0.40, 0.02) == 2
    # Edge: high confidence but tiny margin → roll up
    assert _gate_rollup(0.85, 0.05) == 3
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_classify.py -v
```

- [ ] **Step 3: Implement `src/job_fit/classify.py`**

```python
from job_fit.data.esco import EscoIndex
from job_fit.llm import call_with_tool
from job_fit.models import CVJson, ISCOClassification


CLASSIFY_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "isco_code": {"type": "string", "description": "4-digit ISCO-08 code"},
        "role_label": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "top1_margin": {"type": "number", "minimum": 0, "maximum": 1,
                        "description": "Gap between best and second-best candidate"},
        "alternatives": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["isco_code", "role_label", "confidence", "top1_margin", "alternatives"],
}


SYSTEM_PROMPT = """You are picking the best ISCO-08 4-digit occupation code for a CV.

You will be given:
- A summary of the candidate's most recent and most relevant roles
- A shortlist of candidate occupations (ISCO code + label + keywords) retrieved by keyword search

Rules:
- Choose ONE 4-digit code from the shortlist. If none fits, use the closest.
- confidence = 0.0–1.0 reflecting how clearly the CV matches one specific code.
- top1_margin = how much your top pick beats your second-best (0–1; near 0 = ambiguous).
- alternatives: 1–4 other codes worth considering.
- Return via the classify_cv tool.
"""


def _candidate_query(cv: CVJson, target_role: str | None) -> str:
    if target_role:
        return target_role
    if not cv.roles:
        return ""
    from job_fit.anchor import choose_anchor_role
    anchor = choose_anchor_role(cv.roles)
    return f"{anchor.title}. {anchor.description or ''}"[:300]


def _format_shortlist(candidates) -> str:
    lines = []
    for c in candidates:
        lines.append(f"- {c.isco_code} : {c.label_en} / {c.label_cs} (keywords: {', '.join(c.keywords[:6])})")
    return "\n".join(lines)


def _gate_rollup(confidence: float, top1_margin: float) -> int:
    """Return ISCO level (4, 3, 2) based on confidence + margin."""
    if confidence >= 0.80 and top1_margin >= 0.15:
        return 4
    if confidence >= 0.60:
        return 3
    return 2


def classify_cv(cv: CVJson, target_role: str | None = None,
                esco: EscoIndex | None = None) -> ISCOClassification:
    esco = esco or EscoIndex.load_default()
    query = _candidate_query(cv, target_role)
    shortlist = esco.search(query, k=8)
    if not shortlist:
        # fallback: take a generic spread of major groups
        shortlist = esco.candidates[:8]

    user_msg = (
        f"Candidate query: {query}\n\n"
        f"Shortlist:\n{_format_shortlist(shortlist)}\n\n"
        f"Recent skills: {', '.join(cv.skills[:15])}\n"
        f"Detected language: {cv.detected_language}"
    )

    resp = call_with_tool(
        system=SYSTEM_PROMPT,
        user=user_msg,
        tool_name="classify_cv",
        tool_schema=CLASSIFY_TOOL_SCHEMA,
        max_tokens=500,
    )
    if resp.tool_use is None:
        raise RuntimeError("LLM did not produce classify_cv tool call")

    raw = resp.tool_use
    chosen_4 = raw["isco_code"]
    level = _gate_rollup(raw["confidence"], raw["top1_margin"])
    isco = chosen_4[:level]
    rollup_reason = None
    if level < 4:
        rollup_reason = (f"low classification confidence "
                         f"(conf={raw['confidence']:.2f}, margin={raw['top1_margin']:.2f}); "
                         f"rolled from 4-digit to {level}-digit")
    return ISCOClassification(
        isco_code=isco,
        isco_level=level,
        role_label=raw["role_label"],
        confidence=raw["confidence"],
        top1_margin=raw["top1_margin"],
        alternatives=raw["alternatives"],
        rollup_reason=rollup_reason,
    )
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_classify.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/classify.py tests/test_classify.py
git commit -m "feat(classify): ESCO retrieval + claude pick + confidence-gated rollup"
```

---

### Task 16: LLM-bounded subscores — impact_scope + leadership_ownership_growth

**Files:**
- Create: `src/job_fit/score_soft.py`
- Test: `tests/test_score_soft.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_score_soft.py
from unittest.mock import patch
from job_fit.llm import LlmResponse
from job_fit.score_soft import score_soft


MOCK = LlmResponse(
    text="",
    tool_use={
        "impact_scope": 55,
        "impact_evidence": ["Led 3 product launches", "Improved test pass rate by 20%"],
        "leadership_ownership_growth": {
            "ownership": 14, "leadership": 10, "learning_trajectory": 16,
            "impact_clarity": 13, "stability_execution": 16,
            "evidence": ["Owned regression suite", "Mentored 2 juniors"],
        },
    },
    usage={"input_tokens": 100, "output_tokens": 80, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    cache_hit=False,
)


def test_score_soft_returns_subscores_and_evidence():
    with patch("job_fit.score_soft.call_with_tool", return_value=MOCK):
        impact, leadership, evidence = score_soft("CV text...", anchor_isco="2512")
    assert impact == 55
    assert leadership == 14 + 10 + 16 + 13 + 16  # = 69
    assert "impact_scope" in evidence
    assert "leadership_ownership_growth" in evidence


def test_subscore_clamping():
    # Should clamp values that come back out of range
    bad = LlmResponse(
        text="",
        tool_use={
            "impact_scope": 150,
            "impact_evidence": [],
            "leadership_ownership_growth": {
                "ownership": 25, "leadership": 25, "learning_trajectory": 25,
                "impact_clarity": 25, "stability_execution": 25, "evidence": [],
            },
        },
        usage=MOCK.usage, cache_hit=False,
    )
    with patch("job_fit.score_soft.call_with_tool", return_value=bad):
        impact, leadership, _ = score_soft("CV", anchor_isco="2512")
    assert impact == 100
    assert leadership == 100
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_score_soft.py -v
```

- [ ] **Step 3: Implement `src/job_fit/score_soft.py`**

```python
from job_fit.llm import call_with_tool


SOFT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "impact_scope": {"type": "integer", "minimum": 0, "maximum": 100},
        "impact_evidence": {"type": "array", "items": {"type": "string"}},
        "leadership_ownership_growth": {
            "type": "object",
            "properties": {
                "ownership": {"type": "integer", "minimum": 0, "maximum": 20},
                "leadership": {"type": "integer", "minimum": 0, "maximum": 20},
                "learning_trajectory": {"type": "integer", "minimum": 0, "maximum": 20},
                "impact_clarity": {"type": "integer", "minimum": 0, "maximum": 20},
                "stability_execution": {"type": "integer", "minimum": 0, "maximum": 20},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ownership", "leadership", "learning_trajectory",
                         "impact_clarity", "stability_execution", "evidence"],
        },
    },
    "required": ["impact_scope", "impact_evidence", "leadership_ownership_growth"],
}


SYSTEM_PROMPT = """You are scoring a candidate CV on TWO soft subscores.

1) impact_scope (0–100):
   0–20: vague responsibilities, no measurable outcomes
   21–40: concrete deliverables, no numbers
   41–60: some measurable results (counts, percentages)
   61–80: business-level impact (revenue, cost, reliability, scale)
   81–100: cross-org / multi-million-scale impact
   Return 1–4 evidence quotes from the CV.

2) leadership_ownership_growth (5 dims × 0–20):
   - ownership: task executor (0) → owns features (10) → owns outcomes/budgets (20)
   - leadership: no signal (0) → mentored/coordinated (10) → led people/strategy/hiring (20)
   - learning_trajectory: stagnant (0) → visible progression (10) → repeated upskilling (20)
   - impact_clarity: vague (0) → concrete deliverables (10) → measurable business results (20)
   - stability_execution: unexplained hops (0) → normal (10) → sustained delivery + promotions (20)
   Return 1–6 evidence quotes total.

Rules:
- Score conservatively. Default to mid-range when unclear.
- Evidence must be quotes/paraphrases from the CV; do not invent.
- Return via the score_soft tool.
"""


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def score_soft(redacted_cv_text: str, anchor_isco: str) -> tuple[int, int, dict[str, list[str]]]:
    """Returns (impact_scope, leadership_ownership_growth, evidence_dict)."""
    resp = call_with_tool(
        system=SYSTEM_PROMPT,
        user=f"Anchor occupation: ISCO {anchor_isco}\n\nCV text:\n{redacted_cv_text}",
        tool_name="score_soft",
        tool_schema=SOFT_TOOL_SCHEMA,
        max_tokens=1500,
    )
    if resp.tool_use is None:
        raise RuntimeError("LLM did not produce score_soft tool call")

    raw = resp.tool_use
    impact = _clamp(int(raw["impact_scope"]), 0, 100)
    log = raw["leadership_ownership_growth"]
    leadership_total = sum(_clamp(int(log[k]), 0, 20) for k in (
        "ownership", "leadership", "learning_trajectory", "impact_clarity", "stability_execution"
    ))
    leadership = _clamp(leadership_total, 0, 100)

    evidence = {
        "impact_scope": list(raw.get("impact_evidence", [])),
        "leadership_ownership_growth": list(log.get("evidence", [])),
    }
    return impact, leadership, evidence
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_score_soft.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/score_soft.py tests/test_score_soft.py
git commit -m "feat(score): impact_scope + leadership_ownership_growth (LLM-bounded)"
```

---

### Task 17: Skills_match subscore (with ESCO fallback)

**Files:**
- Create: `src/job_fit/score_skills.py`
- Test: `tests/test_score_skills.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_score_skills.py
from job_fit.score_skills import score_skills_match


def test_skills_match_high_with_overlap():
    cv_skills = ["python", "django", "postgresql", "aws", "docker", "git", "rest api"]
    score, conf = score_skills_match(cv_skills, isco_code="2512")
    assert score >= 70
    assert conf in ("high", "medium")


def test_skills_match_low_when_no_skills():
    score, conf = score_skills_match([], isco_code="2512")
    assert score < 30


def test_skills_match_irrelevant_skills():
    cv_skills = ["watercolor painting", "yoga"]
    score, _ = score_skills_match(cv_skills, isco_code="2512")
    # No software-relevant overlap; score should be low
    assert score < 50
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_score_skills.py -v
```

- [ ] **Step 3: Implement `src/job_fit/score_skills.py`**

```python
"""Skills match subscore.

Two-tier:
- HIGH-confidence path: use ESCO occupation→skill mapping (NOT IMPLEMENTED in v1 core; STRETCH)
- MEDIUM-confidence path: keyword group overlap with ISCO-specific keyword sets
"""
from job_fit.data.esco import EscoIndex


# ISCO 2-digit → expected skill keyword set (rough, deliberately CZ+EN bilingual)
ISCO_SKILL_KEYWORDS: dict[str, set[str]] = {
    "25": {"python", "java", "typescript", "javascript", "go", "rust", "c#", "c++",
           "sql", "postgresql", "mysql", "nosql", "mongodb", "redis",
           "aws", "azure", "gcp", "kubernetes", "docker", "terraform",
           "git", "ci/cd", "rest", "graphql", "grpc",
           "react", "vue", "angular", "next.js", "django", "flask", "fastapi", "spring",
           "ml", "machine learning", "ai", "data science", "pandas", "numpy", "pytorch", "tensorflow",
           "test automation", "playwright", "selenium", "cypress", "pytest", "junit",
           "linux", "bash", "devops", "sre"},
    "22": {"nursing", "patient care", "phlebotomy", "ekg", "icu", "icu care",
           "medication", "anatomy", "physiology", "first aid",
           "ošetřovatelství", "péče o pacienty"},
    "23": {"teaching", "lesson planning", "curriculum", "classroom management",
           "pedagogika", "didaktika"},
    "24": {"accounting", "audit", "financial reporting", "ifrs", "gaap",
           "marketing", "seo", "ppc", "google ads", "analytics",
           "hr", "recruitment", "payroll",
           "sales", "negotiation", "salesforce", "crm",
           "účetnictví", "marketing", "personalistika"},
    "26": {"law", "litigation", "contracts", "civil law", "criminal law",
           "writing", "editing", "publishing"},
    "13": {"leadership", "management", "strategy", "p&l", "hiring", "okrs",
           "stakeholder management", "vedení týmu", "strategie"},
}


def _normalize(skills: list[str]) -> set[str]:
    return {s.strip().lower() for s in skills if s and s.strip()}


def _breadth_score(skills: set[str]) -> float:
    n = len(skills)
    # 0 skills → 0; 5 skills → ~25; 12+ skills → ~40
    if n == 0:
        return 0
    return min(40, int(round(40 * (1 - 1 / (1 + n / 4)))))


def _depth_score(skills: set[str]) -> float:
    DEPTH_MARKERS = {"architecture", "system design", "tech lead", "principal",
                     "mentor", "mentoring", "code review", "scaling",
                     "distributed systems", "microservices", "leadership",
                     "vedení", "architektura"}
    hits = sum(1 for m in DEPTH_MARKERS if any(m in s for s in skills))
    return min(30, hits * 6)


def _match_score(skills: set[str], isco_code: str) -> float:
    expected = ISCO_SKILL_KEYWORDS.get(isco_code[:2], set())
    if not expected:
        return 10  # neutral fallback
    overlap = sum(1 for kw in expected if any(kw in s or s in kw for s in skills))
    if overlap == 0:
        return 0
    # Cap overlap effect at 30 points
    return min(30, int(round(overlap / max(3, len(expected) / 4) * 30)))


def score_skills_match(skills: list[str], isco_code: str) -> tuple[int, str]:
    """Returns (score 0-100, confidence label).

    v1 core uses keyword-group fallback; ESCO occupation→skill mapping is stretch.
    """
    norm = _normalize(skills)
    breadth = _breadth_score(norm)
    depth = _depth_score(norm)
    match = _match_score(norm, isco_code)
    total = int(round(breadth + depth + match))
    confidence = "medium" if isco_code[:2] in ISCO_SKILL_KEYWORDS else "low"
    return min(100, max(0, total)), confidence
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_score_skills.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/score_skills.py tests/test_score_skills.py
git commit -m "feat(score): skills_match with keyword-group fallback"
```

---

### Task 18: ScoreCard assembly + relevant_experience subscore mapping

**Files:**
- Create: `src/job_fit/score.py`
- Test: `tests/test_score.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_score.py
from datetime import date
from job_fit.models import Role, Education
from job_fit.score import (
    relevant_yoe_to_score, assemble_scorecard,
    band_for_total, WEIGHTS,
)


def test_relevant_yoe_to_score_anchors():
    assert relevant_yoe_to_score(0) == 0
    assert relevant_yoe_to_score(2) == 30
    assert relevant_yoe_to_score(5) == 55
    assert relevant_yoe_to_score(10) == 80
    assert relevant_yoe_to_score(15) == 95
    assert relevant_yoe_to_score(25) == 100


def test_band_thresholds():
    assert band_for_total(20) == "Junior"
    assert band_for_total(50) == "Mid"
    assert band_for_total(70) == "Senior"
    assert band_for_total(85) == "Lead/Principal"
    assert band_for_total(97) == "Exec"


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_assemble_scorecard_total_in_range():
    sc = assemble_scorecard(
        relevant_experience=70, skills_match=60,
        impact_scope=55, leadership_ownership_growth=60, education=70,
        confidence_reasons=[], evidence={},
    )
    assert 0 <= sc.total <= 100
    # 0.25*70 + 0.25*60 + 0.20*55 + 0.20*60 + 0.10*70 = 62.5 → 62 or 63
    assert sc.total in (62, 63)


def test_confidence_low_when_many_reasons():
    sc = assemble_scorecard(
        relevant_experience=50, skills_match=50, impact_scope=50,
        leadership_ownership_growth=50, education=50,
        confidence_reasons=["missing dates", "ISCO rolled to 2-digit", "low parse confidence"],
        evidence={},
    )
    assert sc.confidence == "low"
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_score.py -v
```

- [ ] **Step 3: Implement `src/job_fit/score.py`**

```python
from job_fit.models import ScoreCard

WEIGHTS: dict[str, float] = {
    "relevant_experience": 0.25,
    "skills_match": 0.25,
    "impact_scope": 0.20,
    "leadership_ownership_growth": 0.20,
    "education": 0.10,
}

_YOE_ANCHORS = [(0.0, 0), (2.0, 30), (5.0, 55), (10.0, 80), (15.0, 95), (20.0, 100)]


def relevant_yoe_to_score(years: float) -> int:
    if years <= 0:
        return 0
    if years >= 20:
        return 100
    for (y0, s0), (y1, s1) in zip(_YOE_ANCHORS, _YOE_ANCHORS[1:]):
        if y0 <= years <= y1:
            t = (years - y0) / (y1 - y0)
            return int(round(s0 + t * (s1 - s0)))
    return 100


def band_for_total(total: int) -> str:
    if total < 40:
        return "Junior"
    if total < 60:
        return "Mid"
    if total < 80:
        return "Senior"
    if total < 95:
        return "Lead/Principal"
    return "Exec"


def _confidence_label(reasons: list[str]) -> str:
    n = len(reasons)
    if n == 0:
        return "high"
    if n <= 1:
        return "medium"
    return "low"


def assemble_scorecard(
    *,
    relevant_experience: int,
    skills_match: int,
    impact_scope: int,
    leadership_ownership_growth: int,
    education: int,
    confidence_reasons: list[str],
    evidence: dict[str, list[str]],
) -> ScoreCard:
    subs = {
        "relevant_experience": relevant_experience,
        "skills_match": skills_match,
        "impact_scope": impact_scope,
        "leadership_ownership_growth": leadership_ownership_growth,
        "education": education,
    }
    total = int(round(sum(subs[k] * WEIGHTS[k] for k in subs)))
    return ScoreCard(
        relevant_experience=relevant_experience,
        skills_match=skills_match,
        impact_scope=impact_scope,
        leadership_ownership_growth=leadership_ownership_growth,
        education=education,
        total=total,
        band=band_for_total(total),
        confidence=_confidence_label(confidence_reasons),
        confidence_reasons=confidence_reasons,
        evidence=evidence,
    )
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_score.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/score.py tests/test_score.py
git commit -m "feat(score): scorecard assembly + YoE→score + band thresholds"
```

---

### Task 19: LLM growth-action generation

**Files:**
- Create: `src/job_fit/recommend_actions.py`
- Test: `tests/test_recommend_actions.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_recommend_actions.py
from unittest.mock import patch
from job_fit.llm import LlmResponse
from job_fit.recommend_actions import generate_actions


MOCK = LlmResponse(
    text="",
    tool_use={"actions": [
        "Lead a cross-functional initiative for 6 months and document the business impact in numbers",
        "Add 2 architecture-level skills (system design, distributed systems)",
        "Move from individual contributor to mentoring 2 juniors and presenting to stakeholders",
    ]},
    usage={"input_tokens": 100, "output_tokens": 80, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    cache_hit=False,
)


def test_generates_3_to_5_actions():
    with patch("job_fit.recommend_actions.call_with_tool", return_value=MOCK):
        actions = generate_actions(
            redacted_cv_text="...",
            branch="skill_up_within_role",
            subscore_deltas={"skills_match": 15, "impact_scope": 10},
            target_salary=120000,
            current_isco="2512",
        )
    assert 3 <= len(actions) <= 5


def test_handles_role_family_change_branch():
    with patch("job_fit.recommend_actions.call_with_tool", return_value=MOCK):
        actions = generate_actions(
            redacted_cv_text="...",
            branch="role_family_change",
            subscore_deltas={},
            target_salary=200000,
            current_isco="5223",
        )
    assert len(actions) >= 1
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_recommend_actions.py -v
```

- [ ] **Step 3: Implement `src/job_fit/recommend_actions.py`**

```python
from job_fit.llm import call_with_tool


ACTIONS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {"type": "string"},
            "description": "Concrete actions reachable in 6–12 months, referencing CV evidence where possible.",
        },
    },
    "required": ["actions"],
}


SYSTEM_PROMPT = """You generate 3–5 concrete actions a candidate can take to reach a target salary.

Rules:
- Each action: specific, verb-led, 1–2 sentences, referencing CV evidence when relevant.
- Do NOT invent credentials the candidate doesn't have.
- Use a 6–12 month horizon.
- For "role_family_change" branch: actions should describe how to pivot to a higher-paying ISCO group using transferable skills, NOT skill-up within the same role.
- For "stretch_within_role_or_market_change" branch: combine skill-up with company/industry/geography moves.
- For "skill_up_within_role" branch: focus on the specific subscore deltas given.
- Return via the generate_actions tool.
"""


def generate_actions(
    *,
    redacted_cv_text: str,
    branch: str,
    subscore_deltas: dict[str, float],
    target_salary: int,
    current_isco: str,
) -> list[str]:
    user_msg = (
        f"Branch: {branch}\n"
        f"Target salary: {target_salary}\n"
        f"Current ISCO: {current_isco}\n"
        f"Subscore deltas needed: {subscore_deltas}\n\n"
        f"CV (PII-redacted):\n{redacted_cv_text}"
    )
    resp = call_with_tool(
        system=SYSTEM_PROMPT,
        user=user_msg,
        tool_name="generate_actions",
        tool_schema=ACTIONS_TOOL_SCHEMA,
        max_tokens=1000,
    )
    if resp.tool_use is None:
        raise RuntimeError("LLM did not produce generate_actions tool call")
    return list(resp.tool_use["actions"])
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_recommend_actions.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/recommend_actions.py tests/test_recommend_actions.py
git commit -m "feat(recommend): claude-generated growth actions"
```

---

## Phase 4 — Pipeline orchestration

### Task 20: End-to-end pipeline orchestrator

**Files:**
- Create: `src/job_fit/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test (heavily mocked — verifies wiring, not LLM behavior)**

```python
# tests/test_pipeline.py
from datetime import date
from pathlib import Path
from unittest.mock import patch
from job_fit.models import (
    CVJson, Role, Education, ISCOClassification,
)
from job_fit.data.ispv import IspvRecord


CV = CVJson(
    roles=[Role(title="Senior Software Developer", start_date=date(2018, 1, 1),
                end_date=None, is_current=True, description="Built distributed systems")],
    skills=["python", "aws", "kubernetes", "docker", "postgresql", "system design"],
    education=[Education(degree="Master's", field="Computer Science")],
    languages=[], certifications=[],
    detected_language="en", parse_confidence=0.9,
)
CLS = ISCOClassification(
    isco_code="2512", isco_level=4, role_label="Software developer",
    confidence=0.92, top1_margin=0.30, alternatives=["2519"],
)
ISPV = IspvRecord(
    isco_code="2512", isco_level=4,
    d1=50000, q1=70000, median=95000, q3=130000, d9=180000,
    mean=100000, period="1. pololetí 2025", sphere="MZDOVA", count=10000,
)


def test_pipeline_runs_end_to_end(tmp_path: Path):
    fake_pdf = tmp_path / "cv.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")  # extract is mocked anyway

    with patch("job_fit.pipeline.extract_text", return_value="Sample CV"), \
         patch("job_fit.pipeline.parse_cv", return_value=CV), \
         patch("job_fit.pipeline.classify_cv", return_value=CLS), \
         patch("job_fit.pipeline.score_soft", return_value=(70, 65, {"impact_scope": [], "leadership_ownership_growth": []})), \
         patch("job_fit.pipeline.generate_actions", return_value=["Action 1", "Action 2", "Action 3"]), \
         patch("job_fit.pipeline.IspvIndex.load_default") as mock_load:
        mock_load.return_value.lookup_with_rollup.return_value = ISPV
        from job_fit.pipeline import analyze_cv
        result = analyze_cv(fake_pdf, country="CZ")

    assert result.classification.isco_code == "2512"
    assert 0 <= result.score.total <= 100
    assert result.salary.currency == "CZK"
    assert result.salary.point > 0
    assert len(result.growth_plan.actions) >= 3
    assert "duration_ms" in str(result.pipeline_meta)
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/test_pipeline.py -v
```

- [ ] **Step 3: Implement `src/job_fit/pipeline.py`**

```python
import time
from pathlib import Path
from typing import Optional
from loguru import logger

from job_fit.models import (
    CVJson, ISCOClassification, ScoreCard, SalaryRange, GrowthPlan, ResultJson,
)
from job_fit.extract import extract_text
from job_fit.redact import redact_pii, hash_bytes
from job_fit.parse import parse_cv
from job_fit.classify import classify_cv
from job_fit.score_soft import score_soft
from job_fit.score_skills import score_skills_match
from job_fit.score_education import score_education
from job_fit.score import (
    assemble_scorecard, relevant_yoe_to_score, WEIGHTS, band_for_total,
)
from job_fit.yoe import total_yoe, relevant_yoe
from job_fit.salary_math import (
    score_to_percentile, salary_range_from_percentile,
)
from job_fit.recommend import compute_growth_branch, allocate_subscore_deltas
from job_fit.recommend_actions import generate_actions
from job_fit.data.ispv import IspvIndex


def analyze_cv(
    cv_path: Path | str,
    *,
    country: str = "CZ",
    target_role: Optional[str] = None,
) -> ResultJson:
    meta: dict = {"steps": {}}
    t_total = time.perf_counter()
    cv_path = Path(cv_path)

    # 1. Extract
    t = time.perf_counter()
    raw_text = extract_text(cv_path)
    meta["steps"]["extract"] = {"duration_ms": int((time.perf_counter() - t) * 1000)}

    # 2. Redact
    t = time.perf_counter()
    redacted = redact_pii(raw_text)
    file_hash = hash_bytes(cv_path.read_bytes())
    meta["steps"]["redact"] = {"duration_ms": int((time.perf_counter() - t) * 1000), "sha256": file_hash}

    # 3. Parse
    t = time.perf_counter()
    cv: CVJson = parse_cv(redacted)
    meta["steps"]["parse"] = {"duration_ms": int((time.perf_counter() - t) * 1000),
                              "parse_confidence": cv.parse_confidence,
                              "warnings": cv.extraction_warnings}

    # 4. Classify
    t = time.perf_counter()
    cls: ISCOClassification = classify_cv(cv, target_role=target_role)
    meta["steps"]["classify"] = {"duration_ms": int((time.perf_counter() - t) * 1000),
                                 "isco_level": cls.isco_level,
                                 "confidence": cls.confidence}

    # 5. Score subscores
    t = time.perf_counter()
    rel_y = relevant_yoe(cv.roles, anchor_isco=cls.isco_code)
    rel_score = relevant_yoe_to_score(rel_y)
    skills, skills_conf = score_skills_match(cv.skills, isco_code=cls.isco_code)
    edu = score_education(cv.education, anchor_isco=cls.isco_code)
    impact, leadership, evidence = score_soft(redacted, anchor_isco=cls.isco_code)

    confidence_reasons: list[str] = []
    if cls.rollup_reason:
        confidence_reasons.append(cls.rollup_reason)
    if cv.extraction_warnings:
        confidence_reasons.append(f"extraction warnings: {', '.join(cv.extraction_warnings)}")
    if cv.parse_confidence < 0.6:
        confidence_reasons.append(f"low parse confidence ({cv.parse_confidence:.2f})")
    if skills_conf == "low":
        confidence_reasons.append("skills_match used low-confidence fallback (no ISCO keyword set)")

    score: ScoreCard = assemble_scorecard(
        relevant_experience=rel_score,
        skills_match=skills,
        impact_scope=impact,
        leadership_ownership_growth=leadership,
        education=edu,
        confidence_reasons=confidence_reasons,
        evidence=evidence,
    )
    meta["steps"]["score"] = {"duration_ms": int((time.perf_counter() - t) * 1000),
                              "relevant_yoe_years": rel_y,
                              "total_yoe_years": total_yoe(cv.roles)}

    # 6. Salary
    t = time.perf_counter()
    if country != "CZ":
        raise NotImplementedError(f"Country {country} is stretch (Tasks 25–30 in plan). Use CZ.")
    ispv = IspvIndex.load_default()
    record = ispv.lookup_with_rollup(cls.isco_code)
    if record is None:
        # Hard fallback: use overall median bucket
        raise RuntimeError(f"No ISPV data for ISCO {cls.isco_code} or its rollups")

    percentile = score_to_percentile(score.total)
    band = salary_range_from_percentile(
        percentile, score.confidence,
        d1=record.d1, q1=record.q1, median=record.median, q3=record.q3, d9=record.d9,
    )
    salary = SalaryRange(
        point=band.point, low=band.low, high=band.high,
        percentile=band.percentile, percentile_low=band.percentile_low, percentile_high=band.percentile_high,
        currency="CZK", period="month",
        confidence=score.confidence,
        confidence_reasons=confidence_reasons + [f"data: {record.sphere} ISPV {record.period}"],
        isco_code=cls.isco_code, isco_level=cls.isco_level,
        data_source="MPSV ISPV", data_year=2025,
    )
    meta["steps"]["salary"] = {"duration_ms": int((time.perf_counter() - t) * 1000),
                               "percentile": percentile}

    # 7. Recommend
    t = time.perf_counter()
    branch = compute_growth_branch(
        current_salary=salary.point, current_score=score.total,
        d1=record.d1, q1=record.q1, median=record.median, q3=record.q3, d9=record.d9,
    )
    sub_deltas: dict[str, float] = {}
    if branch.score_delta:
        sub_deltas = allocate_subscore_deltas(
            subscores={
                "relevant_experience": score.relevant_experience,
                "skills_match": score.skills_match,
                "impact_scope": score.impact_scope,
                "leadership_ownership_growth": score.leadership_ownership_growth,
                "education": score.education,
            },
            required_total_delta=branch.score_delta,
            weights=WEIGHTS,
            skip={"relevant_experience"},
        )
    actions = generate_actions(
        redacted_cv_text=redacted,
        branch=branch.name,
        subscore_deltas=sub_deltas,
        target_salary=branch.target_salary,
        current_isco=cls.isco_code,
    )
    growth = GrowthPlan(
        branch=branch.name,
        target_salary=branch.target_salary,
        target_percentile=branch.target_percentile,
        required_score=branch.required_score,
        score_delta=branch.score_delta,
        subscore_deltas=sub_deltas,
        message=branch.message,
        actions=actions,
    )
    meta["steps"]["recommend"] = {"duration_ms": int((time.perf_counter() - t) * 1000),
                                  "branch": branch.name}

    meta["total_duration_ms"] = int((time.perf_counter() - t_total) * 1000)
    logger.info("Pipeline done: total={} ms band={} branch={}",
                meta["total_duration_ms"], score.band, branch.name)

    return ResultJson(
        cv=cv, classification=cls, score=score, salary=salary,
        growth_plan=growth, pipeline_meta=meta,
    )
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_pipeline.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/job_fit/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): end-to-end orchestrator"
```

---

### Task 21: Minimal CLI wrapper

**Files:**
- Create: `src/job_fit/cli.py`
- Test: smoke via shell

- [ ] **Step 1: Implement `src/job_fit/cli.py`**

```python
"""Minimal CLI: `uv run job-fit analyze <file> [--country CZ] [--target-role TITLE] [--json]`"""
import argparse
import json
import sys
from pathlib import Path

from job_fit.pipeline import analyze_cv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="job-fit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("file", type=Path)
    p_analyze.add_argument("--country", default="CZ")
    p_analyze.add_argument("--target-role", default=None)
    p_analyze.add_argument("--json", action="store_true", help="Output ResultJson as JSON")
    args = parser.parse_args(argv)

    if args.cmd == "analyze":
        result = analyze_cv(args.file, country=args.country, target_role=args.target_role)
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            sc = result.score
            sl = result.salary
            print(f"Band:     {sc.band} (total {sc.total}/100)")
            print(f"ISCO:     {result.classification.isco_code} — {result.classification.role_label}")
            print(f"Salary:   {sl.point:,} {sl.currency}/{sl.period}  ({sl.low:,}–{sl.high:,}, {sl.confidence})")
            print(f"Branch:   {result.growth_plan.branch}")
            print("Actions:")
            for a in result.growth_plan.actions:
                print(f"  - {a}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Sanity-check via dry-run with mocked pipeline**

```bash
# Just confirm the CLI parses; this requires API key + sample PDF for real run.
uv run python -c "from job_fit.cli import main; import sys; sys.exit(main(['--help']))"
```

Expected: argparse help printed.

- [ ] **Step 3: Commit**

```bash
git add src/job_fit/cli.py
git commit -m "feat(cli): minimal analyze command"
```

---

## Phase 5 — UI

### Task 22: Streamlit app (4 panels per §16 demo path)

**Files:**
- Create: `app/streamlit_app.py`

- [ ] **Step 1: Implement `app/streamlit_app.py`**

```python
"""Streamlit UI for the Job Fit Estimator.

Demo path (§16 of spec):
1. Upload → 2. Parse → 3. Classify → 4. Score → 5. Salary → 6. Growth Plan → 7. Debug
"""
import tempfile
from pathlib import Path

import streamlit as st
from job_fit.pipeline import analyze_cv

st.set_page_config(page_title="Job Fit & Salary Estimator", layout="wide")
st.title("Job Fit & Salary Estimator")
st.caption("CV → seniority score + salary range + +30% growth plan. Czech ISPV anchor; bilingual CZ/EN.")

with st.sidebar:
    st.header("Run settings")
    country = st.selectbox("Country", ["CZ"], help="EU/US/UK are stretch in v1.")
    target_role = st.text_input("Target role (optional)", placeholder="e.g. Senior Data Engineer")
    st.markdown("---")
    st.caption("This is an interview-grade prototype. PII (emails/phones/URLs/birth dates) is redacted before any LLM call.")

uploaded = st.file_uploader("Upload CV (PDF or DOCX)", type=["pdf", "docx"])

if uploaded is not None:
    suffix = "." + uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    with st.spinner("Analyzing CV..."):
        result = analyze_cv(tmp_path, country=country, target_role=target_role or None)

    cv = result.cv
    cls = result.classification
    sc = result.score
    sl = result.salary
    gp = result.growth_plan

    # Top status row
    cols = st.columns(4)
    cols[0].metric("Total score", f"{sc.total}/100", help=f"Band: {sc.band}")
    cols[1].metric("Salary (point)", f"{sl.point:,} {sl.currency}/mo")
    cols[2].metric("Confidence", sc.confidence.upper())
    cols[3].metric("Pipeline", f"{result.pipeline_meta.get('total_duration_ms', 0)} ms")

    st.markdown("---")

    # Panel 1: Score breakdown
    with st.container():
        st.subheader("1. Score breakdown")
        sub = {
            "relevant_experience": sc.relevant_experience,
            "skills_match": sc.skills_match,
            "impact_scope": sc.impact_scope,
            "leadership_ownership_growth": sc.leadership_ownership_growth,
            "education": sc.education,
        }
        st.bar_chart(sub, horizontal=True)
        if sc.confidence_reasons:
            with st.expander("Confidence reasons"):
                for r in sc.confidence_reasons:
                    st.write(f"- {r}")

    # Panel 2: Salary
    with st.container():
        st.subheader("2. Salary range")
        st.write(f"**Point:** {sl.point:,} {sl.currency}/{sl.period}")
        st.write(f"**Range:** {sl.low:,} – {sl.high:,} {sl.currency} (P{sl.percentile_low:.0f}–P{sl.percentile_high:.0f})")
        st.write(f"**Source:** {sl.data_source} {sl.data_year} — ISCO {sl.isco_code} (level {sl.isco_level})")
        with st.expander("Salary confidence reasons"):
            for r in sl.confidence_reasons:
                st.write(f"- {r}")

    # Panel 3: Strengths & gaps
    with st.container():
        st.subheader("3. Strengths & gaps")
        for subscore_name, evidence in (sc.evidence or {}).items():
            if evidence:
                st.markdown(f"**{subscore_name.replace('_', ' ').title()}:**")
                for q in evidence:
                    st.write(f"> {q}")

    # Panel 4: Growth plan
    with st.container():
        st.subheader(f"4. +30% growth plan — branch: `{gp.branch}`")
        st.write(gp.message)
        st.write(f"**Target salary:** {gp.target_salary:,} {sl.currency}")
        if gp.subscore_deltas:
            st.write("**Subscore deltas needed:**")
            for k, v in gp.subscore_deltas.items():
                if v > 0:
                    st.write(f"- {k}: +{v}")
        st.write("**Actions:**")
        for a in gp.actions:
            st.checkbox(a, key=f"action-{hash(a)}")

    # Debug expander
    with st.expander("Debug: pipeline meta + ResultJson"):
        st.json(result.pipeline_meta)
        st.json(result.model_dump(mode="json"))

else:
    st.info("Upload a CV to begin. Try `samples/cvs/mid_dev.pdf`.")
```

- [ ] **Step 2: Sanity launch (no API key needed for syntax check)**

```bash
uv run python -c "import streamlit; from app.streamlit_app import st; print('imports OK')"
```

If launching the actual UI: `uv run streamlit run app/streamlit_app.py` (requires API key).

- [ ] **Step 3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat(ui): streamlit app with 4-panel demo path"
```

---

## Phase 6 — Synthetic CV pack & E2E validation

### Task 23: Author 5 core synthetic CVs

**Files:**
- Create: `samples/cvs/junior_dev_1y.docx`
- Create: `samples/cvs/mid_dev_4y.docx`
- Create: `samples/cvs/senior_dev_8y.docx`
- Create: `samples/cvs/nurse_to_dev_5y_2y.docx`
- Create: `samples/cvs/buzzword_no_evidence.docx`
- Create: `samples/cvs/_generate.py`

- [ ] **Step 1: Write generator script**

```python
# samples/cvs/_generate.py
"""Generate 5 core synthetic CVs as DOCX (§13.1.9 of spec)."""
from pathlib import Path
from docx import Document

OUT = Path(__file__).parent

CVS = {
    "junior_dev_1y.docx": [
        "John Doe",
        "Junior Software Developer",
        "",
        "EXPERIENCE",
        "Junior Developer — TechCo, 09/2024 — present",
        "- Implemented features in Python/Django under senior supervision.",
        "- Wrote unit tests with pytest.",
        "- Participated in code reviews.",
        "",
        "SKILLS",
        "Python, Django, PostgreSQL, Git, pytest, Docker",
        "",
        "EDUCATION",
        "Bachelor's — Computer Science, Charles University (2021–2024)",
    ],
    "mid_dev_4y.docx": [
        "Jane Smith",
        "Software Engineer",
        "",
        "EXPERIENCE",
        "Software Engineer — FinTech Plus, 05/2022 — present",
        "- Owned the payment ingestion service handling 50k tx/day.",
        "- Reduced p95 latency by 35% by introducing async Postgres pool.",
        "- Mentored 1 intern.",
        "Junior Developer — StartupX, 06/2020 — 04/2022",
        "- Built REST APIs in Flask; helped migrate to FastAPI.",
        "- Wrote integration tests; maintained CI pipeline.",
        "",
        "SKILLS",
        "Python, FastAPI, Flask, PostgreSQL, Redis, Docker, AWS, Kubernetes, Git, pytest, GitHub Actions",
        "",
        "EDUCATION",
        "Master's — Software Engineering, CTU Prague (2018–2020)",
    ],
    "senior_dev_8y.docx": [
        "Pavel Novák",
        "Senior Software Engineer",
        "",
        "EXPERIENCE",
        "Tech Lead — Globex, 01/2022 — present",
        "- Led the platform team (5 engineers); owned architecture and roadmap.",
        "- Designed event-driven microservices handling 2M req/day across 4 regions.",
        "- Reduced infrastructure spend by $180k/year through right-sizing.",
        "- Mentored 4 mid-level engineers; introduced design-doc practice.",
        "Senior Engineer — Acme, 03/2018 — 12/2021",
        "- Led migration from monolith to services; cut p99 latency 60%.",
        "- Owned the search subsystem; introduced observability with Prometheus/Grafana.",
        "Software Engineer — Acme, 09/2016 — 02/2018",
        "- Built core ingestion pipelines in Python and Go.",
        "",
        "SKILLS",
        "Python, Go, distributed systems, system design, Kubernetes, Docker, AWS, Terraform, Kafka, PostgreSQL, Redis, microservices, CI/CD, mentoring, architecture",
        "",
        "EDUCATION",
        "Master's — Computer Science, CTU Prague (2014–2016)",
    ],
    "nurse_to_dev_5y_2y.docx": [
        "Anna Veselá",
        "Software Developer (career changer)",
        "",
        "EXPERIENCE",
        "Software Developer — HealthTech, 06/2023 — present",
        "- Build patient-facing web app in TypeScript/Next.js + Python backend.",
        "- Bridge between clinical staff and engineering team.",
        "Registered Nurse — Motol Hospital, 09/2018 — 05/2023",
        "- ICU care; coordinated handoffs across 3 shifts.",
        "- Trained 6 newly-graduated nurses.",
        "",
        "SKILLS",
        "Python, TypeScript, Next.js, PostgreSQL, REST APIs, Git, Docker, patient care (legacy), clinical workflow design",
        "",
        "EDUCATION",
        "Bachelor's — Nursing (2018), Coding bootcamp (2023)",
    ],
    "buzzword_no_evidence.docx": [
        "Max Power",
        "Visionary Tech Leader",
        "",
        "EXPERIENCE",
        "Chief Synergy Officer — VisionCo, 2020 — present",
        "- Drove paradigm-shifting initiatives leveraging cutting-edge synergies.",
        "- Spearheaded transformative outcomes through best-in-class methodologies.",
        "- Enabled strategic alignment across cross-functional touchpoints.",
        "Senior Innovation Catalyst — IdeaWorks, 2017 — 2020",
        "- Orchestrated holistic ecosystems of disruptive thought leadership.",
        "- Empowered stakeholders via next-generation frameworks.",
        "",
        "SKILLS",
        "Leadership, vision, strategy, innovation, transformation, synergy, alignment, empowerment",
        "",
        "EDUCATION",
        "MBA — Famous Business School (2017)",
    ],
}


def main():
    for fname, lines in CVS.items():
        d = Document()
        for line in lines:
            d.add_paragraph(line)
        d.save(OUT / fname)
        print(f"Wrote {fname}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the CVs**

```bash
uv run python samples/cvs/_generate.py
```

Expected: 5 .docx files in `samples/cvs/`.

- [ ] **Step 3: Commit**

```bash
git add samples/cvs/
git commit -m "test(samples): 5 core synthetic CVs (junior/mid/senior/career-changer/buzzword)"
```

---

### Task 24: E2E rank-order validation tests

**Files:**
- Create: `tests/test_e2e_synthetic.py`

- [ ] **Step 1: Write test (skipped without API key)**

```python
# tests/test_e2e_synthetic.py
"""E2E rank-order assertions on the 5 core synthetic CVs.

These tests hit the live LLM, so they require ANTHROPIC_API_KEY in env.
They run slowly (~10–30s per CV due to 3 LLM calls each). Skipped by default
in CI; run locally as: `uv run pytest tests/test_e2e_synthetic.py -v --runslow`.
"""
import os
from pathlib import Path
import pytest
from job_fit.pipeline import analyze_cv

SAMPLES = Path(__file__).parent.parent / "samples" / "cvs"

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="E2E tests need ANTHROPIC_API_KEY",
)


@pytest.fixture(scope="module")
def results():
    out = {}
    for f in ["junior_dev_1y.docx", "mid_dev_4y.docx", "senior_dev_8y.docx",
              "nurse_to_dev_5y_2y.docx", "buzzword_no_evidence.docx"]:
        out[f] = analyze_cv(SAMPLES / f, country="CZ")
    return out


def test_rank_order_dev_career(results):
    j = results["junior_dev_1y.docx"].score.total
    m = results["mid_dev_4y.docx"].score.total
    s = results["senior_dev_8y.docx"].score.total
    assert j < m < s, f"Got junior={j}, mid={m}, senior={s}"


def test_buzzword_penalized(results):
    bz = results["buzzword_no_evidence.docx"].score.total
    mid = results["mid_dev_4y.docx"].score.total
    assert bz < mid, f"Buzzword CV ({bz}) should score lower than mid_dev ({mid})"


def test_career_changer_relevant_yoe_lower_than_total(results):
    r = results["nurse_to_dev_5y_2y.docx"]
    rel_y = r.pipeline_meta["steps"]["score"]["relevant_yoe_years"]
    tot_y = r.pipeline_meta["steps"]["score"]["total_yoe_years"]
    assert rel_y < tot_y


def test_career_changer_anchors_to_dev(results):
    r = results["nurse_to_dev_5y_2y.docx"]
    # Dev ISCO group is 25xx
    assert r.classification.isco_code.startswith("25"), \
        f"Expected dev ISCO (25xx), got {r.classification.isco_code}"


def test_senior_dev_band(results):
    s = results["senior_dev_8y.docx"].score
    assert s.band in ("Senior", "Lead/Principal"), f"Got {s.band}"


def test_all_have_growth_plans(results):
    for name, r in results.items():
        assert len(r.growth_plan.actions) >= 3, f"{name} got {len(r.growth_plan.actions)} actions"
```

- [ ] **Step 2: Run (locally, with API key)**

```bash
ANTHROPIC_API_KEY=sk-... uv run pytest tests/test_e2e_synthetic.py -v
```

Expected: 6 tests pass. If any fail, investigate the offending CV's pipeline output and tune the rubric — but DON'T tighten assertions, the bands are already loose by design.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_synthetic.py
git commit -m "test(e2e): rank-order assertions on synthetic CV pack"
```

---

## Final core task

### Task 25: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# Job Fit & Salary Estimator

> This is an interview-grade prototype, not a production compensation engine.
> The core implementation focuses on Czech ISPV salary anchoring, transparent scoring,
> and bounded LLM explanations. International salary sources, richer ESCO skill matching,
> and FastAPI are implemented only where time permits and otherwise documented as extension points.

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

## Future work

- Region / education / age salary adjustments when public CZ data permits
- NER-based PII redaction
- Real eval harness with expert panel
- ESCO occupation→skill mapping for richer skills_match
- Bias / fairness audit
- CZ-trained ISCO classifier vs. zero-shot LLM benchmark

## Spec

Full design rationale: `docs/specs/2026-05-04-job-fit-estimator-design.md`
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with pipeline, methodology, status, limitations"
```

---

# Stretch Tasks (only if time remains)

### Task 26 (stretch): Polished CLI

- Add `--verbose` flag with rich-formatted output (use `rich` library)
- Add per-step timing display
- Add `--explain` flag that prints just the +30% explanation
- Bundle as a console-scripts entry point

### Task 27 (stretch): FastAPI endpoint

```python
# src/job_fit/api.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
from job_fit.pipeline import analyze_cv

app = FastAPI(title="Job Fit Estimator API", version="0.1.0")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    country: str = Form("CZ"),
    target_role: str | None = Form(None),
):
    suffix = "." + file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    result = analyze_cv(path, country=country, target_role=target_role)
    return JSONResponse(result.model_dump(mode="json"))
```

Run: `uv run uvicorn job_fit.api:app --reload`. Add `tests/test_api.py` with `httpx.AsyncClient`.

### Task 28 (stretch): Eurostat SES integration

- `scripts/fetch_eurostat.py`: `eurostat` PyPI pkg → `data/eurostat/ses.parquet`
- `src/job_fit/data/eurostat.py`: `EurostatIndex` with same `lookup_with_rollup` interface as `IspvIndex`
- Pipeline `salary.py` step gets a `country` dispatcher

### Task 29 (stretch): Full ESCO occupation→skill mapping

- Extend `scripts/build_esco_index.py` to fetch ESCO API `/occupation/{code}/skills` per code
- Cache as `data/esco/occupation_skills.csv`
- Update `score_skills.py` HIGH-confidence path to use this mapping (Jaccard against expected skill set), bumping skills_match confidence from "medium" to "high"

### Task 30 (stretch): 5 additional adversarial synthetic CVs

Generate the remaining 5 CVs from spec §9.1: `qa_automation_5y`, `nurse_10y`,
`exec_head_dept`, `freelancer_overlap`, `missing_dates`. Add corresponding
assertions to `tests/test_e2e_synthetic.py` (e.g., `freelancer_overlap` total_yoe must
not double-count parallel intervals).

---

## Self-Review Checklist (run before declaring done)

- [ ] All `pyproject.toml` dependencies match imports across modules
- [ ] No "TODO" / "TBD" / placeholder strings in source files
- [ ] Streamlit demo path renders all 4 panels per spec §16
- [ ] CLI runs `uv run job-fit analyze --help` cleanly
- [ ] `uv run pytest tests/ --ignore=tests/test_e2e_synthetic.py` passes (unit tests)
- [ ] With API key set, `uv run pytest tests/test_e2e_synthetic.py` passes (E2E)
- [ ] README implementation-status table reflects what actually shipped
- [ ] All "designed/stretch" items in README are truly out of v1, not accidentally half-built
