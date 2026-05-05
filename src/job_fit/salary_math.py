from dataclasses import dataclass
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
