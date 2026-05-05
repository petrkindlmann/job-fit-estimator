from unittest.mock import patch
from job_fit.llm import LlmResponse
from job_fit.parse import parse_cv


MOCK_RESP = LlmResponse(
    text="",
    tool_use={
        "roles": [{"title": "Software Developer", "company": "ACME",
                   "start_date": "2020-01-01", "end_date": "2024-01-01",
                   "is_current": False, "description": "Built things"}],
        "skills": ["python", "sql"],
        "education": [{"degree": "Bachelor's", "field": "Computer Science"}],
        "languages": [{"code": "en", "proficiency": "C1"}],
        "certifications": [],
        "detected_language": "en",
        "parse_confidence": 0.85,
        "extraction_warnings": [],
    },
    usage={"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    cache_hit=False,
)


def test_parse_cv_returns_cvjson():
    with patch("job_fit.parse.call_with_tool", return_value=MOCK_RESP):
        cv = parse_cv("Some CV text...")
    assert len(cv.roles) == 1
    assert cv.roles[0].title == "Software Developer"
    assert "python" in cv.skills
    assert cv.detected_language == "en"


def test_parse_cv_handles_warnings():
    resp = LlmResponse(
        text="",
        tool_use={**MOCK_RESP.tool_use, "extraction_warnings": ["no dates found"], "parse_confidence": 0.4},
        usage=MOCK_RESP.usage, cache_hit=False,
    )
    with patch("job_fit.parse.call_with_tool", return_value=resp):
        cv = parse_cv("Vague CV")
    assert "no dates found" in cv.extraction_warnings
    assert cv.parse_confidence == 0.4
