"""Streamlit UI for the Job Fit Estimator.

Demo path (§16 of spec):
1. Upload → 2. Parse → 3. Classify → 4. Score → 5. Salary → 6. Growth Plan → 7. Debug
"""
import tempfile
from pathlib import Path

import streamlit as st
from job_fit.pipeline import analyze_cv

st.set_page_config(page_title="Job Fit & Salary Estimator", layout="wide")
st.title("Job Fit & Salary Estimator")
st.caption("CV → seniority score + salary range + +30% growth plan. Czech ISPV anchor; bilingual CZ/EN.")

with st.sidebar:
    st.header("Run settings")
    country = st.selectbox("Country", ["CZ"], help="EU/US/UK are stretch in v1.")
    target_role = st.text_input("Target role (optional)", placeholder="e.g. Senior Data Engineer")
    st.markdown("---")
    st.caption("This is an interview-grade prototype. PII (emails/phones/URLs/birth dates) is redacted before any LLM call.")

uploaded = st.file_uploader("Upload CV (PDF or DOCX)", type=["pdf", "docx"])

if uploaded is not None:
    suffix = "." + uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    with st.spinner("Analyzing CV..."):
        result = analyze_cv(tmp_path, country=country, target_role=target_role or None)

    cv = result.cv
    cls = result.classification
    sc = result.score
    sl = result.salary
    gp = result.growth_plan

    # Top status row
    cols = st.columns(4)
    cols[0].metric("Total score", f"{sc.total}/100", help=f"Band: {sc.band}")
    cols[1].metric("Salary (point)", f"{sl.point:,} {sl.currency}/mo")
    cols[2].metric("Confidence", sc.confidence.upper())
    cols[3].metric("Pipeline", f"{result.pipeline_meta.get('total_duration_ms', 0)} ms")

    st.markdown("---")

    # Panel 1: Score breakdown
    with st.container():
        st.subheader("1. Score breakdown")
        sub = {
            "relevant_experience": sc.relevant_experience,
            "skills_match": sc.skills_match,
            "impact_scope": sc.impact_scope,
            "leadership_ownership_growth": sc.leadership_ownership_growth,
            "education": sc.education,
        }
        st.bar_chart(sub, horizontal=True)
        if sc.confidence_reasons:
            with st.expander("Confidence reasons"):
                for r in sc.confidence_reasons:
                    st.write(f"- {r}")

    # Panel 2: Salary
    with st.container():
        st.subheader("2. Salary range")
        st.write(f"**Point:** {sl.point:,} {sl.currency}/{sl.period}")
        st.write(f"**Range:** {sl.low:,} – {sl.high:,} {sl.currency} (P{sl.percentile_low:.0f}–P{sl.percentile_high:.0f})")
        st.write(f"**Source:** {sl.data_source} {sl.data_year} — ISCO {sl.isco_code} (level {sl.isco_level})")
        with st.expander("Salary confidence reasons"):
            for r in sl.confidence_reasons:
                st.write(f"- {r}")

    # Panel 3: Strengths & gaps
    with st.container():
        st.subheader("3. Strengths & gaps")
        for subscore_name, evidence in (sc.evidence or {}).items():
            if evidence:
                st.markdown(f"**{subscore_name.replace('_', ' ').title()}:**")
                for q in evidence:
                    st.write(f"> {q}")

    # Panel 4: Growth plan
    with st.container():
        st.subheader(f"4. +30% growth plan — branch: `{gp.branch}`")
        st.write(gp.message)
        st.write(f"**Target salary:** {gp.target_salary:,} {sl.currency}")
        if gp.subscore_deltas:
            st.write("**Subscore deltas needed:**")
            for k, v in gp.subscore_deltas.items():
                if v > 0:
                    st.write(f"- {k}: +{v}")
        st.write("**Actions:**")
        for a in gp.actions:
            st.checkbox(a, key=f"action-{hash(a)}")

    # Debug expander
    with st.expander("Debug: pipeline meta + ResultJson"):
        st.json(result.pipeline_meta)
        st.json(result.model_dump(mode="json"))

else:
    st.info("Upload a CV to begin. Try `samples/cvs/mid_dev.pdf`.")
