from job_fit.llm import call_with_tool


SOFT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "impact_scope": {"type": "integer", "minimum": 0, "maximum": 100},
        "impact_evidence": {"type": "array", "items": {"type": "string"}},
        "leadership_ownership_growth": {
            "type": "object",
            "properties": {
                "ownership": {"type": "integer", "minimum": 0, "maximum": 20},
                "leadership": {"type": "integer", "minimum": 0, "maximum": 20},
                "learning_trajectory": {"type": "integer", "minimum": 0, "maximum": 20},
                "impact_clarity": {"type": "integer", "minimum": 0, "maximum": 20},
                "stability_execution": {"type": "integer", "minimum": 0, "maximum": 20},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ownership", "leadership", "learning_trajectory",
                         "impact_clarity", "stability_execution", "evidence"],
        },
    },
    "required": ["impact_scope", "impact_evidence", "leadership_ownership_growth"],
}


SYSTEM_PROMPT = """You are scoring a candidate CV on TWO soft subscores.

1) impact_scope (0–100):
   0–20: vague responsibilities, no measurable outcomes
   21–40: concrete deliverables, no numbers
   41–60: some measurable results (counts, percentages)
   61–80: business-level impact (revenue, cost, reliability, scale)
   81–100: cross-org / multi-million-scale impact
   Return 1–4 evidence quotes from the CV.

2) leadership_ownership_growth (5 dims × 0–20):
   - ownership: task executor (0) → owns features (10) → owns outcomes/budgets (20)
   - leadership: no signal (0) → mentored/coordinated (10) → led people/strategy/hiring (20)
   - learning_trajectory: stagnant (0) → visible progression (10) → repeated upskilling (20)
   - impact_clarity: vague (0) → concrete deliverables (10) → measurable business results (20)
   - stability_execution: unexplained hops (0) → normal (10) → sustained delivery + promotions (20)
   Return 1–6 evidence quotes total.

Rules:
- Score conservatively. Default to mid-range when unclear.
- Evidence must be quotes/paraphrases from the CV; do not invent.
- Return via the score_soft tool.
"""


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def score_soft(redacted_cv_text: str, anchor_isco: str) -> tuple[int, int, dict[str, list[str]]]:
    """Returns (impact_scope, leadership_ownership_growth, evidence_dict)."""
    resp = call_with_tool(
        system=SYSTEM_PROMPT,
        user=f"Anchor occupation: ISCO {anchor_isco}\n\nCV text:\n{redacted_cv_text}",
        tool_name="score_soft",
        tool_schema=SOFT_TOOL_SCHEMA,
        max_tokens=1500,
    )
    if resp.tool_use is None:
        raise RuntimeError("LLM did not produce score_soft tool call")

    raw = resp.tool_use
    impact = _clamp(int(raw["impact_scope"]), 0, 100)
    log = raw["leadership_ownership_growth"]
    leadership_total = sum(_clamp(int(log[k]), 0, 20) for k in (
        "ownership", "leadership", "learning_trajectory", "impact_clarity", "stability_execution"
    ))
    leadership = _clamp(leadership_total, 0, 100)

    evidence = {
        "impact_scope": list(raw.get("impact_evidence", [])),
        "leadership_ownership_growth": list(log.get("evidence", [])),
    }
    return impact, leadership, evidence
