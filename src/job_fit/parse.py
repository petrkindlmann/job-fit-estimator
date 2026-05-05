from job_fit.llm import call_with_tool
from job_fit.models import CVJson


PARSE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": ["string", "null"]},
                    "start_date": {"type": ["string", "null"], "description": "ISO date YYYY-MM-DD or null"},
                    "end_date": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "is_current": {"type": "boolean"},
                    "raw_dates": {"type": ["string", "null"]},
                },
                "required": ["title", "is_current"],
            },
        },
        "skills": {"type": "array", "items": {"type": "string"}},
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "field": {"type": ["string", "null"]},
                    "institution": {"type": ["string", "null"]},
                    "start_date": {"type": ["string", "null"]},
                    "end_date": {"type": ["string", "null"]},
                },
                "required": ["degree"],
            },
        },
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "proficiency": {"type": ["string", "null"]},
                },
                "required": ["code"],
            },
        },
        "certifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "issuer": {"type": ["string", "null"]},
                    "year": {"type": ["integer", "null"]},
                },
                "required": ["name"],
            },
        },
        "detected_language": {"type": "string", "enum": ["cs", "en", "other"]},
        "parse_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "extraction_warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["roles", "skills", "education", "languages", "certifications",
                 "detected_language", "parse_confidence", "extraction_warnings"],
}


SYSTEM_PROMPT = """You are extracting structured CV data into a strict JSON schema.

Rules:
- The CV text has had PII redacted (emails, phones, URLs, birth dates show as [REDACTED_*]). Ignore these tokens.
- Extract every distinct role/job. For ongoing roles, set is_current=true and end_date=null.
- Use ISO YYYY-MM-DD for dates. If only month/year, use the first day of the month.
- If a date is missing or unparseable, set null and add a warning to extraction_warnings.
- Skills: include only technical/professional skills mentioned, deduplicated, lowercased.
- detected_language: "cs" if predominantly Czech, "en" if predominantly English, else "other".
- parse_confidence: 0.0–1.0 reflecting how complete and parseable the CV was.
- extraction_warnings: short tags like "no dates found", "low text length", "PDF extraction likely failed".
- Do NOT invent roles, skills, or dates. If unclear, omit and warn.
- Return all results via the parse_cv tool.
"""


def parse_cv(redacted_text: str) -> CVJson:
    resp = call_with_tool(
        system=SYSTEM_PROMPT,
        user=redacted_text,
        tool_name="parse_cv",
        tool_schema=PARSE_TOOL_SCHEMA,
        max_tokens=4000,
    )
    if resp.tool_use is None:
        raise RuntimeError("LLM did not produce parse_cv tool call")
    return CVJson.model_validate(resp.tool_use)
