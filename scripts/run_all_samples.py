"""Run the pipeline on every CV in samples/cvs/ and write the results.

Outputs:
- docs/sample-results/{cv_name}.json   per-CV ResultJson dump
- docs/sample-results/_summary.json    side-by-side comparison data
- docs/results.html                    rendered report (open in browser)

Run:
  ANTHROPIC_API_KEY=sk-... uv run python scripts/run_all_samples.py
  node scripts/render_results_pdf.mjs   # optional: also produce docs/results.pdf
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from job_fit.pipeline import analyze_cv


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "samples" / "cvs"
OUT_DIR = REPO_ROOT / "docs" / "sample-results"
HTML_OUT = REPO_ROOT / "docs" / "results.html"

CVS = [
    "junior_dev_1y.docx",
    "mid_dev_4y.docx",
    "senior_dev_8y.docx",
    "nurse_to_dev_5y_2y.docx",
    "buzzword_no_evidence.docx",
]


def run_one(cv_filename: str) -> dict:
    path = SAMPLES_DIR / cv_filename
    if not path.exists():
        return {"file": cv_filename, "error": f"not found: {path}"}
    started = time.time()
    try:
        result = analyze_cv(path, country="CZ")
        return {
            "file": cv_filename,
            "ok": True,
            "duration_s": round(time.time() - started, 1),
            "result": result.model_dump(mode="json"),
        }
    except Exception as e:
        return {
            "file": cv_filename,
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(limit=4),
            "duration_s": round(time.time() - started, 1),
        }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for cv in CVS:
        print(f"→ Analyzing {cv} ...", flush=True)
        entry = run_one(cv)
        summary.append(entry)
        out_json = OUT_DIR / cv.replace(".docx", ".json").replace(".pdf", ".json")
        out_json.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
        if entry.get("ok"):
            r = entry["result"]
            print(
                f"   score={r['score']['total']} ({r['score']['band']}), "
                f"salary≈{r['salary']['point']:,} {r['salary']['currency']}, "
                f"branch={r['growth_plan']['branch']}, "
                f"took {entry['duration_s']}s"
            )
        else:
            print(f"   FAILED: {entry['error'][:120]}")

    summary_path = OUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote per-CV JSON + summary → {OUT_DIR}")

    from build_results_report import build_html  # noqa: E402

    build_html(summary, HTML_OUT)
    print(f"Wrote dashboard → {HTML_OUT}")


if __name__ == "__main__":
    main()
