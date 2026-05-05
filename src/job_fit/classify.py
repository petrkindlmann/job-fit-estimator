from job_fit.data.esco import EscoIndex
from job_fit.llm import call_with_tool
from job_fit.models import CVJson, ISCOClassification


CLASSIFY_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "isco_code": {"type": "string", "description": "4-digit ISCO-08 code"},
        "role_label": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "top1_margin": {"type": "number", "minimum": 0, "maximum": 1,
                        "description": "Gap between best and second-best candidate"},
        "alternatives": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["isco_code", "role_label", "confidence", "top1_margin", "alternatives"],
}


SYSTEM_PROMPT = """You are picking the best ISCO-08 4-digit occupation code for a CV.

You will be given:
- A summary of the candidate's most recent and most relevant roles
- A shortlist of candidate occupations (ISCO code + label + keywords) retrieved by keyword search

Rules:
- Choose ONE 4-digit code from the shortlist. If none fits, use the closest.
- confidence = 0.0–1.0 reflecting how clearly the CV matches one specific code.
- top1_margin = how much your top pick beats your second-best (0–1; near 0 = ambiguous).
- alternatives: 1–4 other codes worth considering.
- Return via the classify_cv tool.
"""


def _candidate_query(cv: CVJson, target_role: str | None) -> str:
    if target_role:
        return target_role
    if not cv.roles:
        return ""
    from job_fit.anchor import choose_anchor_role
    anchor = choose_anchor_role(cv.roles)
    return f"{anchor.title}. {anchor.description or ''}"[:300]


def _format_shortlist(candidates) -> str:
    lines = []
    for c in candidates:
        lines.append(f"- {c.isco_code} : {c.label_en} / {c.label_cs} (keywords: {', '.join(c.keywords[:6])})")
    return "\n".join(lines)


def _gate_rollup(confidence: float, top1_margin: float) -> int:
    """Return ISCO level (4, 3, 2) based on confidence + margin."""
    if confidence >= 0.80 and top1_margin >= 0.15:
        return 4
    if confidence >= 0.60:
        return 3
    return 2


def classify_cv(cv: CVJson, target_role: str | None = None,
                esco: EscoIndex | None = None) -> ISCOClassification:
    esco = esco or EscoIndex.load_default()
    query = _candidate_query(cv, target_role)
    shortlist = esco.search(query, k=8)
    if not shortlist:
        # fallback: take a generic spread of major groups
        shortlist = esco.candidates[:8]

    user_msg = (
        f"Candidate query: {query}\n\n"
        f"Shortlist:\n{_format_shortlist(shortlist)}\n\n"
        f"Recent skills: {', '.join(cv.skills[:15])}\n"
        f"Detected language: {cv.detected_language}"
    )

    resp = call_with_tool(
        system=SYSTEM_PROMPT,
        user=user_msg,
        tool_name="classify_cv",
        tool_schema=CLASSIFY_TOOL_SCHEMA,
        max_tokens=500,
    )
    if resp.tool_use is None:
        raise RuntimeError("LLM did not produce classify_cv tool call")

    raw = resp.tool_use
    chosen_4 = raw["isco_code"]
    level = _gate_rollup(raw["confidence"], raw["top1_margin"])
    isco = chosen_4[:level]
    rollup_reason = None
    if level < 4:
        rollup_reason = (f"low classification confidence "
                         f"(conf={raw['confidence']:.2f}, margin={raw['top1_margin']:.2f}); "
                         f"rolled from 4-digit to {level}-digit")
    return ISCOClassification(
        isco_code=isco,
        isco_level=level,
        role_label=raw["role_label"],
        confidence=raw["confidence"],
        top1_margin=raw["top1_margin"],
        alternatives=raw["alternatives"],
        rollup_reason=rollup_reason,
    )
