from job_fit.llm import call_with_tool


ACTIONS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {"type": "string"},
            "description": "Concrete actions reachable in 6–12 months, referencing CV evidence where possible.",
        },
    },
    "required": ["actions"],
}


SYSTEM_PROMPT = """You generate 3–5 concrete actions a candidate can take to reach a target salary.

Rules:
- Each action: specific, verb-led, 1–2 sentences, referencing CV evidence when relevant.
- Do NOT invent credentials the candidate doesn't have.
- Use a 6–12 month horizon.
- For "role_family_change" branch: actions should describe how to pivot to a higher-paying ISCO group using transferable skills, NOT skill-up within the same role.
- For "stretch_within_role_or_market_change" branch: combine skill-up with company/industry/geography moves.
- For "skill_up_within_role" branch: focus on the specific subscore deltas given.
- Return via the generate_actions tool.
"""


def generate_actions(
    *,
    redacted_cv_text: str,
    branch: str,
    subscore_deltas: dict[str, float],
    target_salary: int,
    current_isco: str,
) -> list[str]:
    user_msg = (
        f"Branch: {branch}\n"
        f"Target salary: {target_salary}\n"
        f"Current ISCO: {current_isco}\n"
        f"Subscore deltas needed: {subscore_deltas}\n\n"
        f"CV (PII-redacted):\n{redacted_cv_text}"
    )
    resp = call_with_tool(
        system=SYSTEM_PROMPT,
        user=user_msg,
        tool_name="generate_actions",
        tool_schema=ACTIONS_TOOL_SCHEMA,
        max_tokens=1000,
    )
    if resp.tool_use is None:
        raise RuntimeError("LLM did not produce generate_actions tool call")
    return list(resp.tool_use["actions"])
