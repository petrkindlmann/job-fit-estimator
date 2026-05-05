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
