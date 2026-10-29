from datetime import datetime

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from planner.graph import build_graph, new_replan_run, pending_interrupts
from planner.models import EventSource, Plan
from planner.replan import is_fixed, propose_replan
from planner.scheduler import schedule
from planner.validator import check_user_events_unchanged, validate_plan


@pytest.fixture
def planned(store, window):
    """The seed week with its plan applied, as if approved on Monday morning."""
    plan = schedule(store.list_tasks(), store.list_events(), store.get_preferences(), *window)
    store.apply_plan(plan)
    return store


def now_wed(store) -> datetime:
    return datetime(2026, 11, 4, 9, 0, tzinfo=store.get_preferences().tz)


def sessions(store, *, before=None, after=None):
    out = [e for e in store.list_events() if e.source == EventSource.AGENT and not e.completed]
    if before:
        out = [e for e in out if e.start < before]
    if after:
        out = [e for e in out if e.start >= after]
    return out


def mark_past(store, now, missed_id=None):
    """Mark every past session done, except one marked missed."""
    for ev in sessions(store, before=now):
        store.mark_session(ev.id, done=ev.id != missed_id)


def check_valid(store, result):
    fixed = [e for e in store.list_events() if is_fixed(e)]
    assert validate_plan(result.plan, store.list_tasks(), fixed, store.get_preferences()) == []


def test_mark_session_done_and_missed(planned):
    first, second = sessions(planned)[:2]
    before = planned.get_task(first.task_id).done_minutes
    planned.mark_session(first.id, done=True)
    assert planned.get_event(first.id).completed
    assert planned.get_task(first.task_id).done_minutes == before + 120

    planned.mark_session(second.id, done=False)
    assert planned.get_event(second.id) is None
    with pytest.raises(ValueError):
        planned.mark_session(first.id, done=True)
    with pytest.raises(PermissionError):
        planned.mark_session("ev-dentist", done=True)


def test_missed_session_is_rescheduled_without_moving_the_rest(planned, window):
    now = now_wed(planned)
    missed = next(e for e in sessions(planned, before=now) if e.task_id == "exam")
    mark_past(planned, now, missed_id=missed.id)
    future_before = {(e.task_id, e.start) for e in sessions(planned, after=now)}

    result = propose_replan(planned, *window, now)

    assert {(s.task_id, s.start) for s in result.kept} == future_before  # nothing that was fine moved
    assert result.dropped == [] and result.assumed_missed == []
    assert [(s.task_id, s.minutes) for s in result.added] == [("exam", 120)]
    assert all(s.start >= now for s in result.added)
    check_valid(planned, result)


def test_unmarked_past_sessions_count_as_missed(planned, window):
    now = now_wed(planned)
    past = {e.id for e in sessions(planned, before=now)}
    result = propose_replan(planned, *window, now)
    assert set(result.assumed_missed) == past
    check_valid(planned, result)


def test_extra_work_drops_sessions_no_longer_needed(planned, window):
    now = now_wed(planned)
    mark_past(planned, now)
    first_b = next(e for e in planned.list_events() if e.task_id == "client-b" and e.completed)
    # Client B was finished in one long sitting: 240 min total instead of 120.
    planned.update_task("client-b", done_minutes=240)
    future_b = [e.id for e in sessions(planned, after=now) if e.task_id == "client-b"]

    result = propose_replan(planned, *window, now)
    assert first_b.completed and future_b and result.dropped == future_b
    assert all(s.task_id != "client-b" for s in result.plan.sessions)
    check_valid(planned, result)


def test_full_replan_moves_future_sessions_too(planned, window):
    now = now_wed(planned)
    mark_past(planned, now)
    result = propose_replan(planned, *window, now, keep_future=False)
    assert result.kept == []
    assert set(result.dropped) == {e.id for e in sessions(planned, after=now)}
    check_valid(planned, result)


def test_replan_through_graph_updates_calendar_safely(planned, window):
    now = now_wed(planned)
    missed = next(e for e in sessions(planned, before=now) if e.task_id == "exam")
    mark_past(planned, now, missed_id=missed.id)
    before = planned.list_events()
    completed = [e for e in before if e.completed]

    graph = build_graph(planned, llm=None, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "r1"}}
    graph.invoke(new_replan_run(*window, now, run_id="r1"), config)
    [payload] = pending_interrupts(graph, config)
    assert payload["mode"] == "replan" and payload["replan"]["added"] == 1
    assert planned.list_events() == before  # nothing changes before approval

    graph.invoke(Command(resume={"approved": True}), config)
    assert graph.get_state(config).values["status"] == "done"

    after = planned.list_events()
    assert check_user_events_unchanged(before, after) == []
    assert all(e in after for e in completed)
    open_plan = Plan(sessions=[s for s in Plan.model_validate(payload["plan"]).sessions])
    fixed = [e for e in after if is_fixed(e)]
    assert validate_plan(open_plan, planned.list_tasks(), fixed, planned.get_preferences()) == []
    exam = planned.get_task("exam")
    planned_exam = sum(s.minutes for s in open_plan.sessions if s.task_id == "exam")
    assert exam.done_minutes + planned_exam == exam.estimate_minutes
