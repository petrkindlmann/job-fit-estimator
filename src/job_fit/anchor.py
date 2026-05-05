from datetime import date
from job_fit.models import Role
from job_fit.yoe import role_duration_months


def choose_anchor_role(roles: list[Role], today: date | None = None) -> Role:
    today = today or date.today()
    if not roles:
        raise ValueError("no roles to anchor on")
    current = [r for r in roles if r.is_current]
    if current:
        return max(current, key=lambda r: role_duration_months(r, today))
    substantial = [r for r in roles if role_duration_months(r, today) >= 12]
    pool = substantial or roles
    # Prefer most recent end_date (None ranks highest as "still ongoing")
    return max(pool, key=lambda r: r.end_date or today)
