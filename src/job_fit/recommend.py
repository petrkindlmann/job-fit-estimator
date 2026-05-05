from dataclasses import dataclass
from typing import Iterable
from job_fit.salary_math import salary_to_percentile, percentile_to_required_score


@dataclass
class GrowthBranch:
    name: str          # "skill_up_within_role" | "stretch_within_role_or_market_change" | "role_family_change"
    target_salary: int
    target_percentile: float
    message: str
    required_score: int | None = None
    score_delta: int | None = None


def compute_growth_branch(
    current_salary: float,
    *, d1: float, q1: float, median: float, q3: float, d9: float,
    current_score: float | None = None,
    target_multiplier: float = 1.30,
) -> GrowthBranch:
    target = current_salary * target_multiplier
    target_p = salary_to_percentile(target, d1=d1, q1=q1, median=median, q3=q3, d9=d9)

    if target > d9:
        return GrowthBranch(
            name="role_family_change",
            target_salary=int(round(target)),
            target_percentile=target_p,
            message=(
                "+30% target exceeds the top decile of your current ISCO group. "
                "Skill development inside this role is unlikely to deliver +30%. "
                "Consider role-family change, geography, industry, or compensation model."
            ),
        )

    if target_p >= 85:
        return GrowthBranch(
            name="stretch_within_role_or_market_change",
            target_salary=int(round(target)),
            target_percentile=target_p,
            message=(
                "+30% target lands in the top 15% of your current ISCO distribution. "
                "Reaching it usually requires a combination of demonstrated impact + "
                "moving company/industry/geography, not skill-up alone."
            ),
        )

    required = percentile_to_required_score(target_p)
    delta = max(0, required - int(round(current_score or 0)))
    return GrowthBranch(
        name="skill_up_within_role",
        target_salary=int(round(target)),
        target_percentile=target_p,
        required_score=required,
        score_delta=delta,
        message=(
            f"+30% is reachable inside your current ISCO with a score increase of "
            f"~{delta} points (target P{target_p:.0f})."
        ),
    )


def allocate_subscore_deltas(
    subscores: dict[str, float],
    required_total_delta: float,
    weights: dict[str, float],
    skip: Iterable[str],
) -> dict[str, float]:
    """Distribute a required total-score delta across subscores by weighted capacity.
    Skipped subscores (e.g. relevant_experience — can't fast-track YoE) get 0 delta."""
    if required_total_delta <= 0:
        return {k: 0.0 for k in subscores}
    skip_set = set(skip)
    gaps = {k: max(0.0, 100 - v) for k, v in subscores.items() if k not in skip_set}
    weighted_capacity = {k: gaps[k] * weights[k] for k in gaps}
    total_capacity = sum(weighted_capacity.values()) or 1.0
    plan: dict[str, float] = {k: 0.0 for k in subscores}
    for k in gaps:
        share = required_total_delta * (weighted_capacity[k] / total_capacity)
        # Convert score-points-of-total back to subscore-points: divide by weight
        raw = share / weights[k] if weights[k] > 0 else 0.0
        plan[k] = round(min(gaps[k], raw), 1)
    return plan
