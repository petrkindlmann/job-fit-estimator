import hashlib
import re

PII_PATTERNS: list[tuple[str, str]] = [
    # ORDER MATTERS: linkedin before generic url; birth_date before phone (date numbers can look phone-ish).
    ("BIRTH_DATE", r"(?:datum narození|date of birth|born|narozen[aá]?)\s*[:\-]?\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"),
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("LINKEDIN", r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s)]+"),
    ("URL", r"https?://\S+|www\.\S+"),
    # CZ-style 9-digit phones with optional country prefix.
    # Three groups of 3 digits separated by space, hyphen, or dot.
    # Requires either a country prefix OR all 9 digits in three tight groups.
    # Negative lookahead prevents matching MM/YYYY employment dates.
    ("PHONE", r"(?:(?:\+|00)\d{1,3}[\s.-]?)?\d{3}[\s.-]\d{3}[\s.-]\d{3}(?!\d)"),
    # CZ rodné číslo (birth ID) — 6 digits / 3-4 digits
    ("CZ_ID", r"\b\d{6}\s?/\s?\d{3,4}\b"),
]


def redact_pii(text: str) -> str:
    out = text
    for label, pattern in PII_PATTERNS:
        out = re.sub(pattern, f"[REDACTED_{label}]", out, flags=re.IGNORECASE)
    return out


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
