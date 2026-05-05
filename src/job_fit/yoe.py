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
    if role_isco[1:] == anchor_isco[1:]:
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
