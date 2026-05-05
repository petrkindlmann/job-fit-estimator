from unittest.mock import patch
from job_fit.llm import LlmResponse
from job_fit.score_soft import score_soft


MOCK = LlmResponse(
    text="",
    tool_use={
        "impact_scope": 55,
        "impact_evidence": ["Led 3 product launches", "Improved test pass rate by 20%"],
        "leadership_ownership_growth": {
            "ownership": 14, "leadership": 10, "learning_trajectory": 16,
            "impact_clarity": 13, "stability_execution": 16,
            "evidence": ["Owned regression suite", "Mentored 2 juniors"],
        },
    },
    usage={"input_tokens": 100, "output_tokens": 80, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    cache_hit=False,
)


def test_score_soft_returns_subscores_and_evidence():
    with patch("job_fit.score_soft.call_with_tool", return_value=MOCK):
        impact, leadership, evidence = score_soft("CV text...", anchor_isco="2512")
    assert impact == 55
    assert leadership == 14 + 10 + 16 + 13 + 16  # = 69
    assert "impact_scope" in evidence
    assert "leadership_ownership_growth" in evidence


def test_subscore_clamping():
    # Should clamp values that come back out of range
    bad = LlmResponse(
        text="",
        tool_use={
            "impact_scope": 150,
            "impact_evidence": [],
            "leadership_ownership_growth": {
                "ownership": 25, "leadership": 25, "learning_trajectory": 25,
                "impact_clarity": 25, "stability_execution": 25, "evidence": [],
            },
        },
        usage=MOCK.usage, cache_hit=False,
    )
    with patch("job_fit.score_soft.call_with_tool", return_value=bad):
        impact, leadership, _ = score_soft("CV", anchor_isco="2512")
    assert impact == 100
    assert leadership == 100
