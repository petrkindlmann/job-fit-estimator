from pathlib import Path
from job_fit.extract import extract_text

FIX = Path(__file__).parent / "fixtures"


def test_extract_pdf():
    text = extract_text(FIX / "sample.pdf")
    assert "Senior Developer" in text
    assert "Python" in text


def test_extract_docx():
    text = extract_text(FIX / "sample.docx")
    assert "Senior Developer" in text
    assert "Python" in text


def test_extract_unsupported_extension(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    import pytest
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text(p)
