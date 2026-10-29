"""The planning agent as a LangGraph state machine.

    plan mode:   intake -> (clarify -> intake)* -> schedule -> validate
    replan mode: replan -> validate
    then:        validate -> (repair -> validate)* -> approve -> execute | (revise -> intake) | cancelled

clarify and approve are interrupts: the graph pauses there until the user
answers, and a checkpointer lets it resume later, even from another process.
Nothing is written to the calendar before approval.
"""
from __future__ import annotations

import operator
import uuid
from collections import defaultdict
from datetime import date, datetime
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .intake import drafts_to_tasks, parse_request
from .llm import LLMClient
from .models import Event, Plan, Task, Unscheduled
from .replan import is_fixed, propose_replan
from .scheduler import schedule
from .simulator import PlannerStore, session_event_id
from .validator import check_user_events_unchanged, validate_plan

MAX_CLARIFY_ROUNDS = 2
MAX_REVISIONS = 3
MAX_REPAIRS = 2


class PlannerState(TypedDict, total=False):
    # Values stay JSON-friendly (dicts, strings) so any checkpointer can store them.
    mode: str  # plan | replan
    now: str   # replan only: ISO datetime; sessions before this are in the past
    keep_future: bool  # replan only: keep still-needed future sessions in place
    replan: dict       # replan only: kept/added/dropped/assumed_missed summary
    request: str
    run_id: str
    window_start: str
    window_end: str
    clarifications: list[dict]  # [{"questions": [...], "answer": "..."}]
    feedback: list[str]         # reasons the user gave for rejecting earlier plans
    questions: list[str]
    tasks: list[dict]
    plan: dict
    violations: list[dict]
    repairs: int
    created_event_ids: list[str]
    removed_event_ids: list[str]
    status: str  # planning | revising | approved | done | cancelled | failed
    log: Annotated[list[str], operator.add]  # step trace for the UI and evaluation


def new_replan_run(
    start: date, end: date, now: datetime, *, keep_future: bool = True, run_id: str | None = None
) -> PlannerState:
    return {
        "mode": "replan",
        "now": now.isoformat(),
        "keep_future": keep_future,
        "run_id": run_id or uuid.uuid4().hex[:8],
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "log": [],
    }


def new_run(request: str, start: date, end: date, run_id: str | None = None) -> PlannerState:
    return {
        "mode": "plan",
        "request": request,
        "run_id": run_id or uuid.uuid4().hex[:8],
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "clarifications": [],
        "feedback": [],
        "log": [],
    }


def compose_request(state: PlannerState) -> str:
    """The original request plus everything the user has told us since."""
    parts = [state["request"]]
    if state.get("clarifications"):
        parts.append("\nAnswers to your earlier questions (do not ask about these again):")
        for c in state["clarifications"]:
            parts.extend(f"Q: {q}" for q in c["questions"])
            parts.append(f"A: {c['answer']}")
    if state.get("feedback"):
        parts.append("\nThe user rejected the previous plan with this feedback:")
        parts.extend(f"- {f}" for f in state["feedback"])
    return "\n".join(parts)


def pending_interrupts(graph, config: dict) -> list[Any]:
    """Payloads of the interrupts the graph is currently paused on."""
    snapshot = graph.get_state(config)
    return [i.value for task in snapshot.tasks for i in task.interrupts]


def build_graph(
    store: PlannerStore,
    llm: LLMClient,
    *,
    checkpointer=None,
    scheduler: Callable[..., Plan] = schedule,
):
    def window(state: PlannerState) -> tuple[date, date]:
        return date.fromisoformat(state["window_start"]), date.fromisoformat(state["window_end"])

    def tasks_of(state: PlannerState) -> list[Task]:
        return [Task.model_validate(t) for t in state["tasks"]]

    def is_replan(state: PlannerState) -> bool:
        return state.get("mode") == "replan"

    def busy_events(state: PlannerState) -> list[Event]:
        # A replan's plan already contains every open session, so only fixed events count as busy.
        events = store.list_events()
        return [e for e in events if is_fixed(e)] if is_replan(state) else events

    def route_start(state: PlannerState) -> str:
        return "replan" if is_replan(state) else "intake"

    def replan(state: PlannerState) -> dict:
        start, end = window(state)
        result = propose_replan(
            store, start, end, datetime.fromisoformat(state["now"]), keep_future=state.get("keep_future", True)
        )
        summary = {
            "kept": len(result.kept),
            "added": len(result.added),
            "dropped": result.dropped,
            "assumed_missed": result.assumed_missed,
        }
        return {
            "tasks": [t.model_dump(mode="json") for t in store.list_tasks()],
            "plan": result.plan.model_dump(mode="json"),
            "replan": summary,
            "repairs": 0,
            "status": "planning",
            "log": [
                f"replan: kept {summary['kept']}, added {summary['added']}, "
                f"dropped {len(result.dropped)}, assumed missed {len(result.assumed_missed)}"
            ],
        }

    def intake(state: PlannerState) -> dict:
        prefs = store.get_preferences()
        start, end = window(state)
        parsed = parse_request(llm, compose_request(state), prefs=prefs, start=start, end=end, today=start)
        tasks = drafts_to_tasks(parsed, prefs, id_prefix=state["run_id"])
        return {
            "tasks": [t.model_dump(mode="json") for t in tasks],
            "questions": parsed.questions,
            "status": "planning",
            "log": [f"intake: {len(tasks)} tasks, {len(parsed.questions)} questions"],
        }

    def route_after_intake(state: PlannerState) -> str:
        if state.get("questions") and len(state.get("clarifications", [])) < MAX_CLARIFY_ROUNDS:
            return "clarify"
        return "schedule"

    def clarify(state: PlannerState) -> dict:
        answer = interrupt({"kind": "clarify", "questions": state["questions"], "tasks": state["tasks"]})
        round_ = {"questions": state["questions"], "answer": str(answer)}
        return {"clarifications": [*state.get("clarifications", []), round_], "log": ["clarify: user answered"]}

    def plan_week(state: PlannerState) -> dict:
        start, end = window(state)
        plan = scheduler(tasks_of(state), store.list_events(), store.get_preferences(), start, end)
        return {
            "plan": plan.model_dump(mode="json"),
            "repairs": 0,
            "log": [f"schedule: {len(plan.sessions)} sessions, {len(plan.unscheduled)} unscheduled"],
        }

    def validate(state: PlannerState) -> dict:
        plan = Plan.model_validate(state["plan"])
        violations = validate_plan(plan, tasks_of(state), busy_events(state), store.get_preferences())
        return {"violations": [v.model_dump() for v in violations], "log": [f"validate: {len(violations)} violations"]}

    def route_after_validate(state: PlannerState) -> str:
        if not state["violations"]:
            return "approve"
        fixable = all(v["task_id"] for v in state["violations"])
        return "repair" if fixable and state.get("repairs", 0) < MAX_REPAIRS else "fail"

    def repair(state: PlannerState) -> dict:
        """Drop the sessions of tasks with violations and report that time as unscheduled."""
        plan = Plan.model_validate(state["plan"])
        bad = {v["task_id"]: v["code"] for v in state["violations"]}
        dropped: defaultdict[str, int] = defaultdict(int)
        for s in plan.sessions:
            if s.task_id in bad:
                dropped[s.task_id] += s.minutes
        repaired = Plan(
            sessions=[s for s in plan.sessions if s.task_id not in bad],
            unscheduled=plan.unscheduled + [
                Unscheduled(task_id=t, minutes=m, reason=f"removed by validator ({bad[t]})")
                for t, m in dropped.items()
            ],
        )
        return {
            "plan": repaired.model_dump(mode="json"),
            "repairs": state.get("repairs", 0) + 1,
            "log": [f"repair: dropped sessions for {sorted(bad)}"],
        }

    def fail(state: PlannerState) -> dict:
        return {"status": "failed", "log": ["fail: plan still invalid after repair"]}

    def approve(state: PlannerState) -> dict:
        decision = interrupt({
            "kind": "approval",
            "mode": state.get("mode", "plan"),
            "plan": state["plan"],
            "tasks": state["tasks"],
            "replan": state.get("replan"),
        })
        decision = decision if isinstance(decision, dict) else {}
        if decision.get("approved"):
            return {"status": "approved", "log": ["approve: approved"]}
        feedback = str(decision.get("feedback") or "").strip()
        # Free-text revisions go back through the LLM intake, which only exists in plan mode.
        if feedback and not is_replan(state) and len(state.get("feedback", [])) < MAX_REVISIONS:
            return {
                "status": "revising",
                "feedback": [*state.get("feedback", []), feedback],
                "log": ["approve: revision requested"],
            }
        return {"status": "cancelled", "log": ["approve: cancelled"]}

    def route_after_approve(state: PlannerState) -> str:
        return {"approved": "execute", "revising": "intake"}.get(state["status"], END)

    def execute(state: PlannerState) -> dict:
        before = store.list_events()
        plan = Plan.model_validate(state["plan"])
        removed: list[str] = []
        if is_replan(state):
            # Sync open sessions to the plan: kept ones share ids with their events, the rest go.
            wanted = {session_event_id(s) for s in plan.sessions}
            for ev in before:
                if not is_fixed(ev) and ev.id not in wanted:
                    store.delete_event(ev.id)
                    removed.append(ev.id)
        else:
            for task in tasks_of(state):
                store.put_task(task)
        created = store.apply_plan(plan)  # idempotent if this node re-runs
        problems = check_user_events_unchanged(before, store.list_events())
        if problems:
            return {
                "status": "failed",
                "violations": [v.model_dump() for v in problems],
                "log": ["execute: existing appointments changed"],
            }
        return {
            "status": "done",
            "created_event_ids": [e.id for e in created],
            "removed_event_ids": removed,
            "log": [f"execute: created {len(created)}, removed {len(removed)} events"],
        }

    g = StateGraph(PlannerState)
    g.add_node("intake", intake)
    g.add_node("replan", replan)
    g.add_node("clarify", clarify)
    g.add_node("schedule", plan_week)
    g.add_node("validate", validate)
    g.add_node("repair", repair)
    g.add_node("fail", fail)
    g.add_node("approve", approve)
    g.add_node("execute", execute)

    g.add_conditional_edges(START, route_start, ["intake", "replan"])
    g.add_edge("replan", "validate")
    g.add_conditional_edges("intake", route_after_intake, ["clarify", "schedule"])
    g.add_edge("clarify", "intake")
    g.add_edge("schedule", "validate")
    g.add_conditional_edges("validate", route_after_validate, ["approve", "repair", "fail"])
    g.add_edge("repair", "validate")
    g.add_conditional_edges("approve", route_after_approve, ["execute", "intake", END])
    g.add_edge("execute", END)
    g.add_edge("fail", END)
    return g.compile(checkpointer=checkpointer)
