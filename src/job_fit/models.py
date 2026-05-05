from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Role(BaseModel):
    title: str
    company: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    description: Optional[str] = None
    is_current: bool = False
    raw_dates: Optional[str] = None
    isco_code: Optional[str] = None
    isco_confidence: float = 0.0


class Education(BaseModel):
    degree: str           # "Bachelor's", "Master's", "PhD", "High school", "Vocational", "None"
    field: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class Language(BaseModel):
    code: str             # ISO 639-1 best-effort
    proficiency: Optional[str] = None   # "native", "C2", "B2", etc.


class Certification(BaseModel):
    name: str
    issuer: Optional[str] = None
    year: Optional[int] = None


class CVJson(BaseModel):
    roles: list[Role]
    skills: list[str]
    education: list[Education]
    languages: list[Language]
    certifications: list[Certification]
    detected_language: Literal["cs", "en", "other"]
    parse_confidence: float = Field(ge=0.0, le=1.0)
    extraction_warnings: list[str] = []


class ISCOClassification(BaseModel):
    isco_code: str
    isco_level: Literal[2, 3, 4]
    role_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    top1_margin: float = 0.0
    alternatives: list[str] = []
    rollup_reason: Optional[str] = None


class ScoreCard(BaseModel):
    relevant_experience: int = Field(ge=0, le=100)
    skills_match: int = Field(ge=0, le=100)
    impact_scope: int = Field(ge=0, le=100)
    leadership_ownership_growth: int = Field(ge=0, le=100)
    education: int = Field(ge=0, le=100)
    total: int = Field(ge=0, le=100)
    band: Literal["Junior", "Mid", "Senior", "Lead/Principal", "Exec"]
    confidence: Literal["high", "medium", "low"]
    confidence_reasons: list[str] = []
    evidence: dict[str, list[str]] = {}     # subscore_name -> [evidence quotes]


class SalaryRange(BaseModel):
    point: int
    low: int
    high: int
    percentile: float
    percentile_low: float
    percentile_high: float
    currency: Literal["CZK", "EUR", "USD", "GBP"]
    period: Literal["month", "year"]
    confidence: Literal["high", "medium", "low"]
    confidence_reasons: list[str] = []
    isco_code: str
    isco_level: int
    data_source: str
    data_year: int


class GrowthPlan(BaseModel):
    branch: Literal["skill_up_within_role", "stretch_within_role_or_market_change", "role_family_change"]
    target_salary: int
    target_percentile: Optional[float] = None
    required_score: Optional[int] = None
    score_delta: Optional[int] = None
    subscore_deltas: dict[str, float] = {}
    message: str
    actions: list[str]


class ResultJson(BaseModel):
    cv: CVJson
    classification: ISCOClassification
    score: ScoreCard
    salary: SalaryRange
    growth_plan: GrowthPlan
    pipeline_meta: dict = {}    # timings, token usage, cache hits
