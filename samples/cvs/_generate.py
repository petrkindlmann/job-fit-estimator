"""Generate 5 core synthetic CVs as DOCX (§13.1.9 of spec)."""
from pathlib import Path
from docx import Document

OUT = Path(__file__).parent

CVS = {
    "junior_dev_1y.docx": [
        "John Doe",
        "Junior Software Developer",
        "",
        "EXPERIENCE",
        "Junior Developer — TechCo, 09/2024 — present",
        "- Implemented features in Python/Django under senior supervision.",
        "- Wrote unit tests with pytest.",
        "- Participated in code reviews.",
        "",
        "SKILLS",
        "Python, Django, PostgreSQL, Git, pytest, Docker",
        "",
        "EDUCATION",
        "Bachelor's — Computer Science, Charles University (2021–2024)",
    ],
    "mid_dev_4y.docx": [
        "Jane Smith",
        "Software Engineer",
        "",
        "EXPERIENCE",
        "Software Engineer — FinTech Plus, 05/2022 — present",
        "- Owned the payment ingestion service handling 50k tx/day.",
        "- Reduced p95 latency by 35% by introducing async Postgres pool.",
        "- Mentored 1 intern.",
        "Junior Developer — StartupX, 06/2020 — 04/2022",
        "- Built REST APIs in Flask; helped migrate to FastAPI.",
        "- Wrote integration tests; maintained CI pipeline.",
        "",
        "SKILLS",
        "Python, FastAPI, Flask, PostgreSQL, Redis, Docker, AWS, Kubernetes, Git, pytest, GitHub Actions",
        "",
        "EDUCATION",
        "Master's — Software Engineering, CTU Prague (2018–2020)",
    ],
    "senior_dev_8y.docx": [
        "Pavel Novák",
        "Senior Software Engineer",
        "",
        "EXPERIENCE",
        "Tech Lead — Globex, 01/2022 — present",
        "- Led the platform team (5 engineers); owned architecture and roadmap.",
        "- Designed event-driven microservices handling 2M req/day across 4 regions.",
        "- Reduced infrastructure spend by $180k/year through right-sizing.",
        "- Mentored 4 mid-level engineers; introduced design-doc practice.",
        "Senior Engineer — Acme, 03/2018 — 12/2021",
        "- Led migration from monolith to services; cut p99 latency 60%.",
        "- Owned the search subsystem; introduced observability with Prometheus/Grafana.",
        "Software Engineer — Acme, 09/2016 — 02/2018",
        "- Built core ingestion pipelines in Python and Go.",
        "",
        "SKILLS",
        "Python, Go, distributed systems, system design, Kubernetes, Docker, AWS, Terraform, Kafka, PostgreSQL, Redis, microservices, CI/CD, mentoring, architecture",
        "",
        "EDUCATION",
        "Master's — Computer Science, CTU Prague (2014–2016)",
    ],
    "nurse_to_dev_5y_2y.docx": [
        "Anna Veselá",
        "Software Developer (career changer)",
        "",
        "EXPERIENCE",
        "Software Developer — HealthTech, 06/2023 — present",
        "- Build patient-facing web app in TypeScript/Next.js + Python backend.",
        "- Bridge between clinical staff and engineering team.",
        "Registered Nurse — Motol Hospital, 09/2018 — 05/2023",
        "- ICU care; coordinated handoffs across 3 shifts.",
        "- Trained 6 newly-graduated nurses.",
        "",
        "SKILLS",
        "Python, TypeScript, Next.js, PostgreSQL, REST APIs, Git, Docker, patient care (legacy), clinical workflow design",
        "",
        "EDUCATION",
        "Bachelor's — Nursing (2018), Coding bootcamp (2023)",
    ],
    "buzzword_no_evidence.docx": [
        "Max Power",
        "Visionary Tech Leader",
        "",
        "EXPERIENCE",
        "Chief Synergy Officer — VisionCo, 2020 — present",
        "- Drove paradigm-shifting initiatives leveraging cutting-edge synergies.",
        "- Spearheaded transformative outcomes through best-in-class methodologies.",
        "- Enabled strategic alignment across cross-functional touchpoints.",
        "Senior Innovation Catalyst — IdeaWorks, 2017 — 2020",
        "- Orchestrated holistic ecosystems of disruptive thought leadership.",
        "- Empowered stakeholders via next-generation frameworks.",
        "",
        "SKILLS",
        "Leadership, vision, strategy, innovation, transformation, synergy, alignment, empowerment",
        "",
        "EDUCATION",
        "MBA — Famous Business School (2017)",
    ],
}


def main():
    for fname, lines in CVS.items():
        d = Document()
        for line in lines:
            d.add_paragraph(line)
        d.save(OUT / fname)
        print(f"Wrote {fname}")


if __name__ == "__main__":
    main()
