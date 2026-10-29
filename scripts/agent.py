"""Run the planning agent interactively.

    python scripts/agent.py "I have an exam on Friday, two client deadlines and three workouts"
    python scripts/agent.py --thread <id>     # resume a paused run
    python scripts/agent.py --reset           # restore the sample week

    python scripts/agent.py --sessions                    # list work sessions and their ids
    python scripts/agent.py --done <id> [--minutes 90]    # record a session as done
    python scripts/agent.py --missed <id>                 # record a session as missed
    python scripts/agent.py --replan --now "2026-11-04 09:00" [--full]

The calendar lives in data/planner.db and run state in data/checkpoints.db,
so you can quit at a question and resume later with --thread.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

from langgraph.types import Command

from planner.graph import build_graph, new_replan_run, new_run, pending_interrupts
from planner.llm import make_llm
from planner.models import EventSource, Plan, Task
from planner.render import format_plan
from planner.replan import is_fixed
from planner.runtime import checkpoint_path, open_checkpointer, open_store, planning_window, reset_store
from planner.simulator import PlannerStore


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lstrip("﻿")  # piped input on Windows may start with a BOM
    except EOFError:
        return ""


def handle(payload: dict, store: PlannerStore):
    """Show one interrupt to the user and return the value to resume with."""
    if payload["kind"] == "clarify":
        print("\nBefore I plan, a few questions:")
        for q in payload["questions"]:
            print(f"  ? {q}")
        return ask("\nYour answer (one message): ")

    plan = Plan.model_validate(payload["plan"])
    tasks = [Task.model_validate(t) for t in payload["tasks"]]
    events = store.list_events()
    if payload.get("mode") == "replan":
        events = [e for e in events if is_fixed(e)]  # the plan already holds every open session
    print("\nProposed plan:")
    print(format_plan(plan, tasks, events, store.get_preferences()))
    if payload.get("mode") == "replan":
        r = payload["replan"]
        print(f"\nChanges: keeps {r['kept']} sessions, adds {r['added']}, removes {len(r['dropped'])} "
              f"no longer needed and {len(r['assumed_missed'])} past sessions never marked (treated as missed).")
        answer = ask("\nApply these changes? [y = yes, n = cancel]: ")
        return {"approved": answer.lower() in {"y", "yes"}}
    answer = ask("\nApprove and add to calendar? [y = yes, n = cancel, or describe changes]: ")
    if answer.lower() in {"y", "yes"}:
        return {"approved": True}
    if answer.lower() in {"", "n", "no"}:
        return {"approved": False}
    return {"approved": False, "feedback": answer}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", nargs="?", help="what to plan")
    parser.add_argument("--thread", help="resume an existing run")
    parser.add_argument("--provider", help="gemini or ollama (default: LLM_PROVIDER from .env)")
    parser.add_argument("--reset", action="store_true", help="restore the sample week and forget runs")
    parser.add_argument("--sessions", action="store_true", help="list work sessions and their ids")
    parser.add_argument("--done", metavar="ID", help="record a work session as done")
    parser.add_argument("--minutes", type=int, help="with --done: minutes actually worked")
    parser.add_argument("--missed", metavar="ID", help="record a work session as missed")
    parser.add_argument("--replan", action="store_true", help="reschedule unfinished work")
    parser.add_argument("--now", help="with --replan: current time as 'YYYY-MM-DD HH:MM' (default: now)")
    parser.add_argument("--full", action="store_true", help="with --replan: move future sessions too")
    args = parser.parse_args()

    store = open_store()
    if args.reset:
        reset_store(store)
        checkpoint_path().unlink(missing_ok=True)
        print("Calendar restored to the sample week.")
        if not args.request:
            return 0
    prefs = store.get_preferences()

    if args.sessions:
        for ev in store.list_events():
            if ev.source == EventSource.AGENT:
                state = "done" if ev.completed else "planned"
                print(f"  {ev.id}  {ev.start.astimezone(prefs.tz):%a %d %b %H:%M}  {state:7}  {ev.title}")
        return 0
    if args.done or args.missed:
        task = store.mark_session(args.done or args.missed, done=bool(args.done), minutes=args.minutes)
        print(f"Recorded. {task.title}: {task.remaining_minutes} min still to do.")
        return 0
    if not (args.request or args.thread or args.replan):
        parser.error("give a request, --thread to resume, or --replan")

    llm = None if args.replan else make_llm(args.provider)  # replanning needs no LLM
    graph = build_graph(store, llm, checkpointer=open_checkpointer())

    start, end = planning_window()
    if args.thread:
        config = {"configurable": {"thread_id": args.thread}}
    else:
        if args.replan:
            now = (datetime.strptime(args.now, "%Y-%m-%d %H:%M").replace(tzinfo=prefs.tz)
                   if args.now else datetime.now(prefs.tz))
            state = new_replan_run(start, end, now, keep_future=not args.full)
        else:
            state = new_run(args.request, start, end)
        config = {"configurable": {"thread_id": state["run_id"]}}
        print(f"Run {state['run_id']} (resume with --thread {state['run_id']})")
        graph.invoke(state, config)

    while pending := pending_interrupts(graph, config):
        graph.invoke(Command(resume=handle(pending[0], store)), config)

    final = graph.get_state(config).values
    print(f"\nStatus: {final.get('status')}")
    if final.get("created_event_ids") or final.get("removed_event_ids"):
        print(f"Added {len(final.get('created_event_ids', []))} and removed "
              f"{len(final.get('removed_event_ids', []))} events.")
    for v in final.get("violations") or []:
        print(f"  ! {v['code']}: {v['message']}")
    print("Trace: " + " -> ".join(final.get("log", [])))
    for c in llm.calls if llm else []:
        print(f"LLM {c.provider}/{c.model}: {c.latency_s:.2f}s, in={c.input_tokens} out={c.output_tokens}"
              f"{' (cached)' if c.cached else ''}")
    return 0 if final.get("status") in {"done", "cancelled"} else 1


if __name__ == "__main__":
    sys.exit(main())
