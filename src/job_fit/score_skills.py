"""Skills match subscore.

Two-tier:
- HIGH-confidence path: use ESCO occupation→skill mapping (NOT IMPLEMENTED in v1 core; STRETCH)
- MEDIUM-confidence path: keyword group overlap with ISCO-specific keyword sets
"""

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
    """Score 0-40 based on number of skills. Reaches ~35 by 7 skills, caps at 40."""
    n = len(skills)
    if n == 0:
        return 0
    # linear: min(40, n * 6) → n=7 → 42 → 40; n=6 → 36; n=2 → 12
    return min(40, round(n * 6))


def _depth_score(skills: set[str]) -> float:
    DEPTH_MARKERS = {"architecture", "system design", "tech lead", "principal",
                     "mentor", "mentoring", "code review", "scaling",
                     "distributed systems", "microservices", "leadership",
                     "vedení", "architektura"}
    hits = sum(1 for m in DEPTH_MARKERS if any(m in s for s in skills))
    return min(30, hits * 6)


def _match_score(skills: set[str], isco_code: str) -> float:
    """Score 0-30 based on keyword overlap with the ISCO group's expected skill set."""
    expected = ISCO_SKILL_KEYWORDS.get(isco_code[:2], set())
    if not expected:
        return 10  # neutral fallback
    overlap = sum(1 for kw in expected if any(kw in s or s in kw for s in skills))
    if overlap == 0:
        return 0
    # Simpler denominator: 5 — so 7 hits = 7/5*30 = 42 → capped at 30
    return min(30, round(overlap / 5 * 30))


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
