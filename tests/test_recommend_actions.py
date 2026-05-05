from unittest.mock import patch
from job_fit.llm import LlmResponse
from job_fit.recommend_actions import generate_actions


MOCK = LlmResponse(
    text="",
    tool_use={"actions": [
        "Lead a cross-functional initiative for 6 months and document the business impact in numbers",
        "Add 2 architecture-level skills (system design, distributed systems)",
        "Move from individual contributor to mentoring 2 juniors and presenting to stakeholders",
    ]},
    usage={"input_tokens": 100, "output_tokens": 80, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    cache_hit=False,
)


def test_generates_3_to_5_actions():
    with patch("job_fit.recommend_actions.call_with_tool", return_value=MOCK):
        actions = generate_actions(
            redacted_cv_text="...",
            branch="skill_up_within_role",
            subscore_deltas={"skills_match": 15, "impact_scope": 10},
            target_salary=120000,
            current_isco="2512",
        )
    assert 3 <= len(actions) <= 5


def test_handles_role_family_change_branch():
    with patch("job_fit.recommend_actions.call_with_tool", return_value=MOCK):
        actions = generate_actions(
            redacted_cv_text="...",
            branch="role_family_change",
            subscore_deltas={},
            target_salary=200000,
            current_isco="5223",
        )
    assert len(actions) >= 1
