from job_fit.models import Education
from job_fit.score_education import score_education


def test_no_education():
    assert score_education([], anchor_isco="2511") == 30


def test_phd_unrelated_decreases():
    # PhD with unrelated field, anchor is software dev
    edu = [Education(degree="PhD", field="Philosophy")]
    s = score_education(edu, anchor_isco="2511")  # 90 - 20 = 70
    assert s == 70


def test_phd_related_increases():
    edu = [Education(degree="PhD", field="Computer Science")]
    s = score_education(edu, anchor_isco="2511")  # 90 + 10, capped to 100
    assert s == 100


def test_bachelor_neutral():
    edu = [Education(degree="Bachelor's", field="Marketing")]
    s = score_education(edu, anchor_isco="2511")
    assert s == 65 - 20  # unrelated penalty


def test_picks_highest_degree():
    edu = [
        Education(degree="High school"),
        Education(degree="Master's", field="Computer Science"),
    ]
    s = score_education(edu, anchor_isco="2511")
    assert s == 80 + 10  # master's + relevance bump
