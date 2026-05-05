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
