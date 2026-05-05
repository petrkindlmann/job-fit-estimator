from datetime import date
import pytest
from job_fit.models import Role
from job_fit.yoe import total_yoe, relevant_yoe, isco_similarity, role_duration_months


TODAY = date(2026, 5, 4)


def role(title, start=None, end=None, isco=None, conf=1.0, current=False):
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
