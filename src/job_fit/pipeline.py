import time
from pathlib import Path
from typing import Optional
from loguru import logger

from job_fit.models import (
    CVJson, ISCOClassification, ScoreCard, SalaryRange, GrowthPlan, ResultJson,
)
from job_fit.extract import extract_text
from job_fit.redact import redact_pii, hash_bytes
from job_fit.parse import parse_cv
from job_fit.classify import classify_cv
from job_fit.score_soft import score_soft
from job_fit.score_skills import score_skills_match
from job_fit.score_education import score_education
from job_fit.score import (
    assemble_scorecard, relevant_yoe_to_score, WEIGHTS, band_for_total,
)
from job_fit.yoe import total_yoe, relevant_yoe
from job_fit.salary_math import (
    score_to_percentile, salary_range_from_percentile,
)
from job_fit.recommend import compute_growth_branch, allocate_subscore_deltas
from job_fit.recommend_actions import generate_actions
from job_fit.data.ispv import IspvIndex


def analyze_cv(
    cv_path: Path | str,
    *,
    country: str = "CZ",
    target_role: Optional[str] = None,
) -> ResultJson:
    meta: dict = {"steps": {}}
    t_total = time.perf_counter()
    cv_path = Path(cv_path)

    # 1. Extract
    t = time.perf_counter()
    raw_text = extract_text(cv_path)
    meta["steps"]["extract"] = {"duration_ms": int((time.perf_counter() - t) * 1000)}

    # 2. Redact
    t = time.perf_counter()
    redacted = redact_pii(raw_text)
    file_hash = hash_bytes(cv_path.read_bytes())
    meta["steps"]["redact"] = {"duration_ms": int((time.perf_counter() - t) * 1000), "sha256": file_hash}

    # 3. Parse
    t = time.perf_counter()
    cv: CVJson = parse_cv(redacted)
    meta["steps"]["parse"] = {"duration_ms": int((time.perf_counter() - t) * 1000),
                              "parse_confidence": cv.parse_confidence,
                              "warnings": cv.extraction_warnings}

    # 4. Classify
    t = time.perf_counter()
    cls: ISCOClassification = classify_cv(cv, target_role=target_role)
    meta["steps"]["classify"] = {"duration_ms": int((time.perf_counter() - t) * 1000),
                                 "isco_level": cls.isco_level,
                                 "confidence": cls.confidence}

    # 5. Score subscores
    t = time.perf_counter()
    rel_y = relevant_yoe(cv.roles, anchor_isco=cls.isco_code)
    rel_score = relevant_yoe_to_score(rel_y)
    skills, skills_conf = score_skills_match(cv.skills, isco_code=cls.isco_code)
    edu = score_education(cv.education, anchor_isco=cls.isco_code)
    impact, leadership, evidence = score_soft(redacted, anchor_isco=cls.isco_code)

    confidence_reasons: list[str] = []
    if cls.rollup_reason:
        confidence_reasons.append(cls.rollup_reason)
    if cv.extraction_warnings:
        confidence_reasons.append(f"extraction warnings: {', '.join(cv.extraction_warnings)}")
    if cv.parse_confidence < 0.6:
        confidence_reasons.append(f"low parse confidence ({cv.parse_confidence:.2f})")
    if skills_conf == "low":
        confidence_reasons.append("skills_match used low-confidence fallback (no ISCO keyword set)")

    score: ScoreCard = assemble_scorecard(
        relevant_experience=rel_score,
        skills_match=skills,
        impact_scope=impact,
        leadership_ownership_growth=leadership,
        education=edu,
        confidence_reasons=confidence_reasons,
        evidence=evidence,
    )
    meta["steps"]["score"] = {"duration_ms": int((time.perf_counter() - t) * 1000),
                              "relevant_yoe_years": rel_y,
                              "total_yoe_years": total_yoe(cv.roles)}

    # 6. Salary
    t = time.perf_counter()
    if country != "CZ":
        raise NotImplementedError(f"Country {country} is stretch (Tasks 25–30 in plan). Use CZ.")
    ispv = IspvIndex.load_default()
    record = ispv.lookup_with_rollup(cls.isco_code)
    if record is None:
        raise RuntimeError(f"No ISPV data for ISCO {cls.isco_code} or its rollups")

    percentile = score_to_percentile(score.total)
    band = salary_range_from_percentile(
        percentile, score.confidence,
        d1=record.d1, q1=record.q1, median=record.median, q3=record.q3, d9=record.d9,
    )
    salary = SalaryRange(
        point=band.point, low=band.low, high=band.high,
        percentile=band.percentile, percentile_low=band.percentile_low, percentile_high=band.percentile_high,
        currency="CZK", period="month",
        confidence=score.confidence,
        confidence_reasons=confidence_reasons + [f"data: {record.sphere} ISPV {record.period}"],
        isco_code=cls.isco_code, isco_level=cls.isco_level,
        data_source="MPSV ISPV", data_year=2025,
    )
    meta["steps"]["salary"] = {"duration_ms": int((time.perf_counter() - t) * 1000),
                               "percentile": percentile}

    # 7. Recommend
    t = time.perf_counter()
    branch = compute_growth_branch(
        current_salary=salary.point, current_score=score.total,
        d1=record.d1, q1=record.q1, median=record.median, q3=record.q3, d9=record.d9,
    )
    sub_deltas: dict[str, float] = {}
    if branch.score_delta:
        sub_deltas = allocate_subscore_deltas(
            subscores={
                "relevant_experience": score.relevant_experience,
                "skills_match": score.skills_match,
                "impact_scope": score.impact_scope,
                "leadership_ownership_growth": score.leadership_ownership_growth,
                "education": score.education,
            },
            required_total_delta=branch.score_delta,
            weights=WEIGHTS,
            skip={"relevant_experience"},
        )
    actions = generate_actions(
        redacted_cv_text=redacted,
        branch=branch.name,
        subscore_deltas=sub_deltas,
        target_salary=branch.target_salary,
        current_isco=cls.isco_code,
    )
    growth = GrowthPlan(
        branch=branch.name,
        target_salary=branch.target_salary,
        target_percentile=branch.target_percentile,
        required_score=branch.required_score,
        score_delta=branch.score_delta,
        subscore_deltas=sub_deltas,
        message=branch.message,
        actions=actions,
    )
    meta["steps"]["recommend"] = {"duration_ms": int((time.perf_counter() - t) * 1000),
                                  "branch": branch.name}

    meta["total_duration_ms"] = int((time.perf_counter() - t_total) * 1000)
    logger.info("Pipeline done: total={} ms band={} branch={}",
                meta["total_duration_ms"], score.band, branch.name)

    return ResultJson(
        cv=cv, classification=cls, score=score, salary=salary,
        growth_plan=growth, pipeline_meta=meta,
    )
