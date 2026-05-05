"""Build docs/results.html from the per-CV outputs in docs/sample-results/.

Stand-alone (no external deps) so it can be regenerated from cached JSONs without
re-hitting the LLM. Called by run_all_samples.py at the end of a fresh run, or
directly via `uv run python scripts/build_results_report.py`.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "sample-results"
HTML_OUT = REPO_ROOT / "docs" / "results.html"


CSS = """
:root { --ink:#1a1a1a; --ink-soft:#4a4a4a; --rule:#e2e2dc; --paper:#fdfdfb;
        --accent:#2563eb; --accent-soft:#eef2ff; --good:#047857; --good-soft:#ecfdf5;
        --warn:#b45309; --warn-soft:#fef3c7; --code:#f6f6f2; --mono:'SF Mono',Menlo,monospace; }
* { box-sizing: border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; color:var(--ink);
       background:var(--paper); max-width:1080px; margin:0 auto; padding:48px 32px 80px; line-height:1.55; }
header { border-bottom:2px solid var(--ink); padding-bottom:20px; margin-bottom:32px; }
h1 { margin:0 0 6px; font-size:2rem; letter-spacing:-0.01em; }
.subtitle { color:var(--ink-soft); margin:0; }
h2 { font-size:1.4rem; margin:48px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--rule); }
h3 { font-size:1.1rem; margin:24px 0 8px; }
table { border-collapse:collapse; width:100%; margin:12px 0 18px; font-size:0.93rem; }
th, td { border-bottom:1px solid var(--rule); padding:8px 10px; text-align:left; vertical-align:top; }
th { background:var(--code); font-weight:600; }
.num { text-align:right; font-variant-numeric:tabular-nums; font-family:var(--mono); }
.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.72rem; font-weight:600;
        letter-spacing:0.02em; }
.pill-good { background:var(--good-soft); color:var(--good); }
.pill-warn { background:var(--warn-soft); color:var(--warn); }
.pill-info { background:var(--accent-soft); color:var(--accent); }
.pill-fail { background:#fee2e2; color:#b91c1c; }
.cv-card { border:1px solid var(--rule); border-radius:6px; padding:18px 22px; margin:18px 0;
           background:white; }
.cv-card h3 { margin:0 0 6px; font-size:1.1rem; }
.cv-meta { color:var(--ink-soft); font-size:0.85rem; margin-bottom:12px; }
.score-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:14px 0; }
.score-cell { background:var(--code); border-radius:4px; padding:8px 10px; }
.score-cell .label { font-size:0.7rem; color:var(--ink-soft); text-transform:uppercase;
                     letter-spacing:0.04em; }
.score-cell .value { font-size:1.3rem; font-weight:600; font-variant-numeric:tabular-nums;
                     font-family:var(--mono); }
.bar-track { background:var(--code); height:8px; border-radius:4px; overflow:hidden; margin:4px 0; }
.bar-fill { background:var(--accent); height:100%; }
.summary-row { display:grid; grid-template-columns:1.5fr 0.7fr 0.7fr 1.2fr 1fr 1fr; gap:10px;
               align-items:center; padding:10px 0; border-bottom:1px solid var(--rule); font-size:0.9rem; }
.summary-row.head { font-weight:600; background:var(--code); padding:10px; border-radius:4px;
                    margin-bottom:4px; border:none; }
.evidence { color:var(--ink-soft); font-size:0.88rem; padding:4px 0 4px 16px; border-left:3px solid var(--rule);
            margin:4px 0; }
.actions { margin:8px 0 0; padding-left:22px; }
.actions li { margin-bottom:8px; font-size:0.92rem; }
.detail { margin-top:10px; }
.detail summary { cursor:pointer; color:var(--accent); font-size:0.88rem; }
.detail pre { background:var(--code); border:1px solid var(--rule); padding:10px 14px; border-radius:4px;
              font-family:var(--mono); font-size:0.78rem; overflow-x:auto; white-space:pre-wrap;
              word-break:break-word; max-height:380px; overflow-y:auto; }
.confidence { font-size:0.8rem; color:var(--ink-soft); margin-top:4px; }
.error-box { background:#fef2f2; border:1px solid #fecaca; padding:14px 18px; border-radius:4px;
             margin-top:8px; }
.error-box pre { background:transparent; border:none; padding:0; font-size:0.78rem; color:#b91c1c; }
.empty-state { padding:24px; background:var(--warn-soft); border:1px solid #fde68a; border-radius:6px;
               margin-top:24px; }
@media print {
  body { max-width:none; padding:18px 24px; font-size:10.5pt; }
  .cv-card { page-break-inside:avoid; break-inside:avoid; }
  .actions li { page-break-inside:avoid; }
  pre { page-break-inside:avoid; max-height:none !important; }
}
"""


def _bar(value: int, max_value: int = 100) -> str:
    pct = max(0, min(100, value / max_value * 100))
    return f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%"></div></div>'


def _band_pill(band: str) -> str:
    cls = {
        "Junior": "pill-warn",
        "Mid": "pill-info",
        "Senior": "pill-good",
        "Lead/Principal": "pill-good",
        "Exec": "pill-good",
    }.get(band, "pill-info")
    return f'<span class="pill {cls}">{html.escape(band)}</span>'


def _confidence_pill(conf: str) -> str:
    cls = {"high": "pill-good", "medium": "pill-info", "low": "pill-warn"}.get(conf, "pill-info")
    return f'<span class="pill {cls}">{html.escape(conf)}</span>'


def _branch_pill(branch: str) -> str:
    cls = {
        "skill_up_within_role": "pill-good",
        "stretch_within_role_or_market_change": "pill-info",
        "role_family_change": "pill-warn",
    }.get(branch, "pill-info")
    return f'<span class="pill {cls}">{html.escape(branch)}</span>'


def _format_money(amount: int, currency: str, period: str) -> str:
    return f"{amount:,} {html.escape(currency)}/{html.escape(period[:2])}"


def _render_cv_card(entry: dict) -> str:
    fname = entry["file"]
    if not entry.get("ok"):
        err = entry.get("error", "unknown error")
        trace = html.escape(entry.get("trace", "")[:1500])
        return f"""
<div class="cv-card">
  <h3>{html.escape(fname)} <span class="pill pill-fail">FAILED</span></h3>
  <div class="cv-meta">duration {entry.get('duration_s', 0)}s</div>
  <div class="error-box">
    <strong>{html.escape(err)}</strong>
    <pre>{trace}</pre>
  </div>
</div>"""

    r = entry["result"]
    score = r["score"]
    salary = r["salary"]
    growth = r["growth_plan"]
    classification = r["classification"]
    cv = r["cv"]
    meta = r.get("pipeline_meta", {})

    score_grid = "".join(
        f'<div class="score-cell"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{score[key]}</div>{_bar(score[key])}</div>'
        for label, key in [
            ("rel. exp", "relevant_experience"),
            ("skills", "skills_match"),
            ("impact", "impact_scope"),
            ("lead/own", "leadership_ownership_growth"),
            ("education", "education"),
        ]
    )

    impact_evidence = "".join(
        f'<div class="evidence">{html.escape(q)}</div>'
        for q in (score.get("evidence") or {}).get("impact_scope", [])
    )
    leader_evidence = "".join(
        f'<div class="evidence">{html.escape(q)}</div>'
        for q in (score.get("evidence") or {}).get("leadership_ownership_growth", [])
    )

    actions = "".join(f"<li>{html.escape(a)}</li>" for a in growth.get("actions", []))
    deltas = growth.get("subscore_deltas") or {}
    delta_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>+{v:.1f}</td></tr>"
        for k, v in deltas.items() if v and v > 0
    ) or "<tr><td colspan='2' style='color:var(--ink-soft)'>(branch not skill_up — no subscore deltas)</td></tr>"

    confidence_reasons = "".join(
        f"<li>{html.escape(r)}</li>"
        for r in (score.get("confidence_reasons") or [])
    )

    duration_s = round(meta.get("total_duration_ms", 0) / 1000, 1)

    raw_json = html.escape(json.dumps(r, indent=2, ensure_ascii=False))

    return f"""
<div class="cv-card">
  <h3>{html.escape(fname)} &nbsp; {_band_pill(score['band'])} &nbsp; {_confidence_pill(score['confidence'])}</h3>
  <div class="cv-meta">
    classified as <strong>{html.escape(classification['role_label'])}</strong>
    (ISCO <code>{html.escape(classification['isco_code'])}</code>, level {classification['isco_level']},
    confidence {classification['confidence']:.2f}) &nbsp;·&nbsp;
    pipeline took {duration_s}s &nbsp;·&nbsp;
    {len(cv['roles'])} role(s) parsed, {len(cv['skills'])} skills
  </div>

  <h4 style="margin:14px 0 4px;font-size:0.78rem;color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.04em;">Score breakdown — total {score['total']}/100</h4>
  <div class="score-grid">{score_grid}</div>

  <h4 style="margin:18px 0 4px;font-size:0.78rem;color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.04em;">Salary estimate</h4>
  <table>
    <tr>
      <td>Point estimate</td><td class="num">{_format_money(salary['point'], salary['currency'], salary['period'])}</td>
      <td>at percentile</td><td class="num">P{salary['percentile']:.0f}</td>
    </tr>
    <tr>
      <td>Range</td><td class="num">{salary['low']:,} – {salary['high']:,} {salary['currency']}</td>
      <td>P{salary['percentile_low']:.0f} – P{salary['percentile_high']:.0f}</td><td>{_confidence_pill(salary['confidence'])}</td>
    </tr>
    <tr>
      <td>Source</td><td colspan="3">{html.escape(salary['data_source'])} {salary['data_year']} · ISCO <code>{html.escape(salary['isco_code'])}</code> (level {salary['isco_level']})</td>
    </tr>
  </table>
  {f'<div class="confidence">Confidence reasons: <ul>{confidence_reasons}</ul></div>' if confidence_reasons else ''}

  <h4 style="margin:18px 0 4px;font-size:0.78rem;color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.04em;">Evidence quoted by the LLM</h4>
  {f'<div><strong style="font-size:0.85rem">impact_scope</strong>{impact_evidence}</div>' if impact_evidence else ''}
  {f'<div style="margin-top:8px"><strong style="font-size:0.85rem">leadership_ownership_growth</strong>{leader_evidence}</div>' if leader_evidence else ''}

  <h4 style="margin:18px 0 4px;font-size:0.78rem;color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.04em;">+30% growth plan {_branch_pill(growth['branch'])}</h4>
  <p style="margin:4px 0 8px;font-size:0.92rem;">{html.escape(growth.get('message', ''))}</p>
  <table style="font-size:0.88rem;width:auto;">
    <tr><td>Target salary</td><td class="num">{_format_money(growth['target_salary'], salary['currency'], salary['period'])}</td></tr>
    {f"<tr><td>Target percentile</td><td class='num'>P{growth['target_percentile']:.0f}</td></tr>" if growth.get('target_percentile') else ''}
    {f"<tr><td>Required score</td><td class='num'>{growth['required_score']}</td></tr>" if growth.get('required_score') else ''}
    {f"<tr><td>Score delta</td><td class='num'>+{growth['score_delta']}</td></tr>" if growth.get('score_delta') else ''}
  </table>
  <h4 style="margin:14px 0 4px;font-size:0.78rem;color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.04em;">Subscore deltas needed</h4>
  <table style="font-size:0.88rem;max-width:380px">{delta_rows}</table>

  <h4 style="margin:14px 0 4px;font-size:0.78rem;color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.04em;">Recommended actions ({len(growth.get('actions', []))})</h4>
  <ol class="actions">{actions}</ol>

  <details class="detail">
    <summary>Full ResultJson</summary>
    <pre>{raw_json}</pre>
  </details>
</div>
"""


def _render_summary_table(summary: list[dict]) -> str:
    rows = []
    rows.append(
        '<div class="summary-row head">'
        '<div>CV</div><div>Score</div><div>Band</div><div>Salary</div>'
        '<div>Branch</div><div>ISCO</div></div>'
    )
    for entry in summary:
        if not entry.get("ok"):
            rows.append(
                f'<div class="summary-row">'
                f'<div><code>{html.escape(entry["file"])}</code></div>'
                f'<div colspan="5"><span class="pill pill-fail">FAILED</span> '
                f'{html.escape(entry.get("error","")[:80])}</div></div>'
            )
            continue
        r = entry["result"]
        rows.append(
            f'<div class="summary-row">'
            f'<div><code>{html.escape(entry["file"])}</code></div>'
            f'<div class="num">{r["score"]["total"]}/100</div>'
            f'<div>{_band_pill(r["score"]["band"])}</div>'
            f'<div class="num">{_format_money(r["salary"]["point"], r["salary"]["currency"], r["salary"]["period"])}</div>'
            f'<div>{_branch_pill(r["growth_plan"]["branch"])}</div>'
            f'<div><code>{html.escape(r["classification"]["isco_code"])}</code> '
            f'{html.escape(r["classification"]["role_label"][:24])}</div>'
            f'</div>'
        )
    return "\n".join(rows)


def build_html(summary: list[dict], out_path: Path) -> None:
    if not summary:
        body = '<div class="empty-state"><strong>No results yet.</strong> Run <code>uv run python scripts/run_all_samples.py</code> with <code>ANTHROPIC_API_KEY</code> set.</div>'
    else:
        ok_count = sum(1 for e in summary if e.get("ok"))
        body = (
            f'<p style="color:var(--ink-soft)">{ok_count}/{len(summary)} CVs analysed successfully.</p>'
            f'<h2>Side-by-side summary</h2>'
            + _render_summary_table(summary)
            + '<h2>Per-CV detail</h2>'
            + "\n".join(_render_cv_card(e) for e in summary)
        )

    out_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Job Fit Estimator — Sample Results</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>Job Fit Estimator — Sample Results</h1>
  <p class="subtitle">Live pipeline output on the 5 synthetic CVs in <code>samples/cvs/</code></p>
</header>
{body}
<footer style="margin-top:60px;padding-top:18px;border-top:1px solid var(--rule);
              color:#888;font-size:0.85rem">
  Generated by <code>scripts/build_results_report.py</code> from
  <code>docs/sample-results/_summary.json</code>.
  Re-run <code>uv run python scripts/run_all_samples.py</code> to refresh.
</footer>
</body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    summary_path = RESULTS_DIR / "_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    build_html(summary, HTML_OUT)
    print(f"Wrote {HTML_OUT} ({len(summary)} entries)")


if __name__ == "__main__":
    main()
