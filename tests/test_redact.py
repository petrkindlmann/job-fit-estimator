from job_fit.redact import redact_pii, hash_bytes


def test_redact_email():
    text = "Contact: john.doe@example.com for info"
    out = redact_pii(text)
    assert "john.doe@example.com" not in out
    assert "[REDACTED_EMAIL]" in out


def test_redact_phone_intl():
    text = "Call +420 777 123 456 anytime"
    out = redact_pii(text)
    assert "777 123 456" not in out
    assert "[REDACTED_PHONE]" in out


def test_redact_url_and_linkedin():
    text = "Profile https://linkedin.com/in/johndoe and site www.example.com"
    out = redact_pii(text)
    assert "linkedin.com/in/johndoe" not in out
    assert "[REDACTED_LINKEDIN]" in out or "[REDACTED_URL]" in out


def test_redact_birth_date_cs():
    text = "Datum narození: 12.03.1990"
    out = redact_pii(text)
    assert "12.03.1990" not in out
    assert "[REDACTED_BIRTH_DATE]" in out


def test_redact_does_not_strip_employment_dates():
    text = "Worked at ACME from 03/2018 to 06/2022 as engineer"
    out = redact_pii(text)
    assert "03/2018" in out and "06/2022" in out


def test_hash_bytes_stable():
    h1 = hash_bytes(b"hello")
    h2 = hash_bytes(b"hello")
    assert h1 == h2 and len(h1) == 64
