import os
from dataclasses import dataclass
from typing import Any
import anthropic
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


@dataclass
class LlmResponse:
    text: str
    tool_use: dict[str, Any] | None
    usage: dict[str, int]
    cache_hit: bool


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _model() -> str:
    return os.environ.get("MODEL_ID", "claude-sonnet-4-6")


def call_with_tool(
    *,
    system: str,
    user: str,
    tool_name: str,
    tool_schema: dict[str, Any],
    cache_system: bool = True,
    max_tokens: int = 2000,
) -> LlmResponse:
    """Single-tool call. Forces tool_use; returns parsed tool input.

    `cache_system=True` enables Anthropic prompt caching on the system block —
    used heavily because rubrics are reused across CVs.
    """
    sys_block: list[dict[str, Any]] = [{"type": "text", "text": system}]
    if cache_system:
        sys_block[0]["cache_control"] = {"type": "ephemeral"}

    client = _client()
    resp = client.messages.create(
        model=_model(),
        max_tokens=max_tokens,
        system=sys_block,
        tools=[{"name": tool_name, "input_schema": tool_schema, "description": "Single result tool."}],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )

    tool_input: dict[str, Any] | None = None
    text_parts: list[str] = []
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            tool_input = block.input
        elif block.type == "text":
            text_parts.append(block.text)

    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
    }
    cache_hit = usage["cache_read_input_tokens"] > 0
    logger.debug("LLM call: tool={} usage={} cache_hit={}", tool_name, usage, cache_hit)

    return LlmResponse(
        text="\n".join(text_parts),
        tool_use=tool_input,
        usage=usage,
        cache_hit=cache_hit,
    )
