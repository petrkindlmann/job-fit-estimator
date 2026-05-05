from unittest.mock import patch
from job_fit.llm import LlmResponse
from job_fit.models import CVJson, Role
from job_fit.classify import classify_cv, _gate_rollup


def cv(*, role_title="Senior Software Developer"):
    return CVJson(
        roles=[Role(title=role_title, is_current=True)],
        skills=["python"],
        education=[],
        languages=[],
        certifications=[],
        detected_language="en",
        parse_confidence=0.9,
    )


HIGH_CONF_RESP = LlmResponse(
    text="",
    tool_use={"isco_code": "2512", "role_label": "Software developer",
              "confidence": 0.92, "top1_margin": 0.30,
              "alternatives": ["2511", "2519"]},
    usage={"input_tokens": 100, "output_tokens": 30, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    cache_hit=False,
)


def test_classify_high_confidence_keeps_4_digit():
    with patch("job_fit.classify.call_with_tool", return_value=HIGH_CONF_RESP):
        c = classify_cv(cv())
    assert c.isco_code == "2512"
    assert c.isco_level == 4


def test_classify_rolls_up_when_low_confidence():
    low = LlmResponse(
        text="",
        tool_use={"isco_code": "2512", "role_label": "Software developer",
                  "confidence": 0.55, "top1_margin": 0.05,
                  "alternatives": ["2511", "2519", "2434"]},
        usage=HIGH_CONF_RESP.usage, cache_hit=False,
    )
    with patch("job_fit.classify.call_with_tool", return_value=low):
        c = classify_cv(cv())
    assert c.isco_level == 2  # rolled up to 2-digit
    assert c.isco_code == "25"
    assert c.rollup_reason is not None


def test_gate_rollup_thresholds():
    assert _gate_rollup(0.90, 0.20) == 4
    assert _gate_rollup(0.70, 0.10) == 3
    assert _gate_rollup(0.40, 0.02) == 2
    # Edge: high confidence but tiny margin → roll up
    assert _gate_rollup(0.85, 0.05) == 3
