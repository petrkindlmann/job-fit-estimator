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
