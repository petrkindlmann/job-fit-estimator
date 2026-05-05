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
