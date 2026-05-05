from datetime import date
from job_fit.models import Role, CVJson, ISCOClassification, ScoreCard, SalaryRange, GrowthPlan, ResultJson


def test_role_minimum():
    r = Role(title="Dev", is_current=True)
    assert r.title == "Dev"
    assert r.start_date is None
    assert r.isco_code is None
    assert r.isco_confidence == 0.0


def test_cvjson_with_warnings():
    cv = CVJson(
        roles=[Role(title="Dev", is_current=True)],
        skills=["python"],
        education=[],
        languages=[],
        certifications=[],
        detected_language="en",
        parse_confidence=0.8,
        extraction_warnings=["no dates found"],
    )
    assert cv.parse_confidence == 0.8
    assert "no dates found" in cv.extraction_warnings


def test_scorecard_band_values():
    sc = ScoreCard(
        relevant_experience=70, skills_match=70, impact_scope=60,
        leadership_ownership_growth=60, education=70,
        total=66, band="Senior", confidence="high", confidence_reasons=[],
    )
    assert sc.band == "Senior"


def test_salary_range_percentile_fields():
    sr = SalaryRange(
        point=80000, low=70000, high=90000,
        percentile=60.0, percentile_low=50.0, percentile_high=70.0,
        currency="CZK", period="month", confidence="high",
        confidence_reasons=["CZ ISCO-4 match"],
        isco_code="2511", isco_level=4,
        data_source="MPSV ISPV", data_year=2025,
    )
    assert sr.point == 80000
