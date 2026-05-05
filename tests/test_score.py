from datetime import date
from job_fit.models import Role, Education
from job_fit.score import (
    relevant_yoe_to_score, assemble_scorecard,
    band_for_total, WEIGHTS,
)


def test_relevant_yoe_to_score_anchors():
    assert relevant_yoe_to_score(0) == 0
    assert relevant_yoe_to_score(2) == 30
    assert relevant_yoe_to_score(5) == 55
    assert relevant_yoe_to_score(10) == 80
    assert relevant_yoe_to_score(15) == 95
    assert relevant_yoe_to_score(25) == 100


def test_band_thresholds():
    assert band_for_total(20) == "Junior"
    assert band_for_total(50) == "Mid"
    assert band_for_total(70) == "Senior"
    assert band_for_total(85) == "Lead/Principal"
    assert band_for_total(97) == "Exec"


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_assemble_scorecard_total_in_range():
    sc = assemble_scorecard(
        relevant_experience=70, skills_match=60,
        impact_scope=55, leadership_ownership_growth=60, education=70,
        confidence_reasons=[], evidence={},
    )
    assert 0 <= sc.total <= 100
    # 0.25*70 + 0.25*60 + 0.20*55 + 0.20*60 + 0.10*70 = 62.5 → 62 or 63
    assert sc.total in (62, 63)


def test_confidence_low_when_many_reasons():
    sc = assemble_scorecard(
        relevant_experience=50, skills_match=50, impact_scope=50,
        leadership_ownership_growth=50, education=50,
        confidence_reasons=["missing dates", "ISCO rolled to 2-digit", "low parse confidence"],
        evidence={},
    )
    assert sc.confidence == "low"
