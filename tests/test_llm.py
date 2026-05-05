import os
import pytest
from job_fit.llm import call_with_tool


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no API key in env")
def test_smoke_tool_call():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    resp = call_with_tool(
        system="You answer in one word.",
        user="What is the capital of France?",
        tool_name="answer",
        tool_schema=schema,
        max_tokens=200,
    )
    assert resp.tool_use is not None
    assert "paris" in resp.tool_use["answer"].lower()
