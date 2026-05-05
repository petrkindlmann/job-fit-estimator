"""E2E rank-order assertions on the 5 core synthetic CVs.

These tests hit the live LLM, so they require ANTHROPIC_API_KEY in env.
They run slowly (~10–30s per CV due to 3 LLM calls each). Skipped by default
in CI; run locally as: `uv run pytest tests/test_e2e_synthetic.py -v --runslow`.
"""
import os
from pathlib import Path
import pytest
from job_fit.pipeline import analyze_cv

SAMPLES = Path(__file__).parent.parent / "samples" / "cvs"

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="E2E tests need ANTHROPIC_API_KEY",
)


@pytest.fixture(scope="module")
def results():
    out = {}
    for f in ["junior_dev_1y.docx", "mid_dev_4y.docx", "senior_dev_8y.docx",
              "nurse_to_dev_5y_2y.docx", "buzzword_no_evidence.docx"]:
        out[f] = analyze_cv(SAMPLES / f, country="CZ")
    return out


def test_rank_order_dev_career(results):
    j = results["junior_dev_1y.docx"].score.total
    m = results["mid_dev_4y.docx"].score.total
    s = results["senior_dev_8y.docx"].score.total
    assert j < m < s, f"Got junior={j}, mid={m}, senior={s}"


def test_buzzword_penalized(results):
    bz = results["buzzword_no_evidence.docx"].score.total
    mid = results["mid_dev_4y.docx"].score.total
    assert bz < mid, f"Buzzword CV ({bz}) should score lower than mid_dev ({mid})"


def test_career_changer_relevant_yoe_lower_than_total(results):
    r = results["nurse_to_dev_5y_2y.docx"]
    rel_y = r.pipeline_meta["steps"]["score"]["relevant_yoe_years"]
    tot_y = r.pipeline_meta["steps"]["score"]["total_yoe_years"]
    assert rel_y < tot_y


def test_career_changer_anchors_to_dev(results):
    r = results["nurse_to_dev_5y_2y.docx"]
    # Dev ISCO group is 25xx
    assert r.classification.isco_code.startswith("25"), \
        f"Expected dev ISCO (25xx), got {r.classification.isco_code}"


def test_senior_dev_band(results):
    s = results["senior_dev_8y.docx"].score
    assert s.band in ("Senior", "Lead/Principal"), f"Got {s.band}"


def test_all_have_growth_plans(results):
    for name, r in results.items():
        assert len(r.growth_plan.actions) >= 3, f"{name} got {len(r.growth_plan.actions)} actions"
