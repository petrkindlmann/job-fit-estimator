from datetime import date
from pathlib import Path
from unittest.mock import patch
from job_fit.models import (
    CVJson, Role, Education, ISCOClassification,
)
from job_fit.data.ispv import IspvRecord


CV = CVJson(
    roles=[Role(title="Senior Software Developer", start_date=date(2018, 1, 1),
                end_date=None, is_current=True, description="Built distributed systems")],
    skills=["python", "aws", "kubernetes", "docker", "postgresql", "system design"],
    education=[Education(degree="Master's", field="Computer Science")],
    languages=[], certifications=[],
    detected_language="en", parse_confidence=0.9,
)
CLS = ISCOClassification(
    isco_code="2512", isco_level=4, role_label="Software developer",
    confidence=0.92, top1_margin=0.30, alternatives=["2519"],
)
ISPV = IspvRecord(
    isco_code="2512", isco_level=4,
    d1=50000, q1=70000, median=95000, q3=130000, d9=180000,
    mean=100000, period="1. pololetí 2025", sphere="MZDOVA", count=10000,
)


def test_pipeline_runs_end_to_end(tmp_path: Path):
    fake_pdf = tmp_path / "cv.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")  # extract is mocked anyway

    with patch("job_fit.pipeline.extract_text", return_value="Sample CV"), \
         patch("job_fit.pipeline.parse_cv", return_value=CV), \
         patch("job_fit.pipeline.classify_cv", return_value=CLS), \
         patch("job_fit.pipeline.score_soft", return_value=(70, 65, {"impact_scope": [], "leadership_ownership_growth": []})), \
         patch("job_fit.pipeline.generate_actions", return_value=["Action 1", "Action 2", "Action 3"]), \
         patch("job_fit.pipeline.IspvIndex.load_default") as mock_load:
        mock_load.return_value.lookup_with_rollup.return_value = ISPV
        from job_fit.pipeline import analyze_cv
        result = analyze_cv(fake_pdf, country="CZ")

    assert result.classification.isco_code == "2512"
    assert 0 <= result.score.total <= 100
    assert result.salary.currency == "CZK"
    assert result.salary.point > 0
    assert len(result.growth_plan.actions) >= 3
    assert "duration_ms" in str(result.pipeline_meta)
