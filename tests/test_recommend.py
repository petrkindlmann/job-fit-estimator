from job_fit.recommend import compute_growth_branch, allocate_subscore_deltas


DECILES = dict(d1=40000, q1=55000, median=75000, q3=105000, d9=150000)


def test_branch_role_family_when_target_above_d9():
    # 130k current * 1.3 = 169k > d9 (150k)
    branch = compute_growth_branch(current_salary=130000, **DECILES)
    assert branch.name == "role_family_change"


def test_branch_stretch_when_target_at_p85():
    # 95k * 1.3 = 123.5k → between q3 and d9 → P~82–86 (depends)
    branch = compute_growth_branch(current_salary=95000, **DECILES)
    # Should be stretch or skill_up; if target_p >= 85 then stretch
    assert branch.name in ("stretch_within_role_or_market_change", "skill_up_within_role")
    if branch.target_percentile >= 85:
        assert branch.name == "stretch_within_role_or_market_change"


def test_branch_skill_up_when_target_in_range():
    # 50k * 1.3 = 65k → P~37
    branch = compute_growth_branch(current_salary=50000, **DECILES)
    assert branch.name == "skill_up_within_role"
    assert branch.target_percentile < 85


def test_allocate_skips_relevant_experience():
    subscores = {
        "relevant_experience": 50, "skills_match": 60,
        "impact_scope": 70, "leadership_ownership_growth": 50, "education": 75,
    }
    weights = {
        "relevant_experience": 0.25, "skills_match": 0.25,
        "impact_scope": 0.20, "leadership_ownership_growth": 0.20, "education": 0.10,
    }
    deltas = allocate_subscore_deltas(subscores, required_total_delta=10, weights=weights,
                                      skip={"relevant_experience"})
    assert deltas["relevant_experience"] == 0
    assert sum(deltas.values()) > 0


def test_allocate_zero_delta_returns_zeros():
    subscores = {"a": 50, "b": 60}
    weights = {"a": 0.5, "b": 0.5}
    deltas = allocate_subscore_deltas(subscores, 0, weights, skip=set())
    assert all(v == 0 for v in deltas.values())
