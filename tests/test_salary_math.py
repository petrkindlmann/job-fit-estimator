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
