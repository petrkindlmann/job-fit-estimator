"""Minimal CLI: `uv run job-fit analyze <file> [--country CZ] [--target-role TITLE] [--json]`"""
import argparse
import json
import sys
from pathlib import Path

from job_fit.pipeline import analyze_cv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="job-fit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("file", type=Path)
    p_analyze.add_argument("--country", default="CZ")
    p_analyze.add_argument("--target-role", default=None)
    p_analyze.add_argument("--json", action="store_true", help="Output ResultJson as JSON")
    args = parser.parse_args(argv)

    if args.cmd == "analyze":
        result = analyze_cv(args.file, country=args.country, target_role=args.target_role)
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            sc = result.score
            sl = result.salary
            print(f"Band:     {sc.band} (total {sc.total}/100)")
            print(f"ISCO:     {result.classification.isco_code} — {result.classification.role_label}")
            print(f"Salary:   {sl.point:,} {sl.currency}/{sl.period}  ({sl.low:,}–{sl.high:,}, {sl.confidence})")
            print(f"Branch:   {result.growth_plan.branch}")
            print("Actions:")
            for a in result.growth_plan.actions:
                print(f"  - {a}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
