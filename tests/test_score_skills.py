from job_fit.score_skills import score_skills_match


def test_skills_match_high_with_overlap():
    cv_skills = ["python", "django", "postgresql", "aws", "docker", "git", "rest api"]
    score, conf = score_skills_match(cv_skills, isco_code="2512")
    assert score >= 70
    assert conf in ("high", "medium")


def test_skills_match_low_when_no_skills():
    score, conf = score_skills_match([], isco_code="2512")
    assert score < 30


def test_skills_match_irrelevant_skills():
    cv_skills = ["watercolor painting", "yoga"]
    score, _ = score_skills_match(cv_skills, isco_code="2512")
    # No software-relevant overlap; score should be low
    assert score < 50
