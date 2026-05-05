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
# Kept as a documented constant for future use.
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

    # +10 if directly relevant to anchor occupation
    # -20 if unrelated to anchor occupation and not specifically required for this role
    if _is_relevant(top.field, anchor_isco):
        base += 10
    else:
        base -= 20

    return max(0, min(100, base))
