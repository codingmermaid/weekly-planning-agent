"""Plan the seeded week, optionally parsing a free-text request with the LLM first.

    python scripts/demo.py
    python scripts/demo.py --request "I have an exam on Friday, two client deadlines and three workouts"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from planner.intake import drafts_to_tasks, parse_request
from planner.llm import make_llm
from planner.render import format_plan
from planner.scheduler import schedule
from planner.simulator import PlannerStore
from planner.validator import validate_plan

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=ROOT / "data" / "seed_week.json", type=Path)
    parser.add_argument("--request", help="free-text request; replaces the seeded task list")
    parser.add_argument("--provider", help="gemini or ollama (default: LLM_PROVIDER from .env)")
    parser.add_argument("--strategy", choices=["balanced", "earliest"], default="balanced")
    args = parser.parse_args()

    window = json.loads(args.seed.read_text(encoding="utf-8"))["window"]
    start, end = date.fromisoformat(window["start"]), date.fromisoformat(window["end"])
    store = PlannerStore.from_seed(args.seed)
    prefs = store.get_preferences()
    tasks = store.list_tasks()

    llm = None
    if args.request:
        llm = make_llm(args.provider)
        parsed = parse_request(llm, args.request, prefs=prefs, start=start, end=end, today=start)
        tasks = drafts_to_tasks(parsed, prefs)
        print("Parsed tasks:")
        for t in tasks:
            print(f"  - {t.title} ({t.kind.value}, {t.estimate_minutes} min, deadline {t.deadline})")
        for q in parsed.questions:
            print(f"  ? {q}")

    events = store.list_events()
    plan = schedule(tasks, events, prefs, start, end, strategy=args.strategy)
    print(format_plan(plan, tasks, events, prefs))

    violations = validate_plan(plan, tasks, events, prefs)
    print(f"\nValidator: {'OK' if not violations else f'{len(violations)} violation(s)'}")
    for v in violations:
        print(f"  ! {v.code}: {v.message}")

    if llm is not None:
        for c in llm.calls:
            print(f"LLM {c.provider}/{c.model}: {c.latency_s:.2f}s, in={c.input_tokens} out={c.output_tokens}"
                  f"{' (cached)' if c.cached else ''}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
