"""Replanning: reschedule unfinished work while keeping what is already settled.

Fixed: the user's appointments and completed sessions. Kept: future sessions
that are still needed stay exactly where they are (minimal change). Only the
shortfall (missed work, or work never planned) goes into free time after `now`.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime

from pydantic import BaseModel, Field

from .models import Event, EventSource, Plan, Session
from .scheduler import schedule
from .simulator import PlannerStore


class ReplanResult(BaseModel):
    plan: Plan  # every open session after replanning: kept + added
    kept: list[Session] = Field(default_factory=list)
    added: list[Session] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)         # future sessions no longer needed
    assumed_missed: list[str] = Field(default_factory=list)  # past sessions never marked done or missed


def is_fixed(event: Event) -> bool:
    """Events a replan never changes: the user's own appointments and completed work."""
    return event.source == EventSource.USER or event.completed


def propose_replan(
    store: PlannerStore, start: date, end: date, now: datetime, *, keep_future: bool = True
) -> ReplanResult:
    prefs = store.get_preferences()
    tasks = {t.id: t for t in store.list_tasks()}
    events = store.list_events()
    fixed = [e for e in events if is_fixed(e)]
    open_sessions = sorted((e for e in events if not is_fixed(e)), key=lambda e: e.start)

    # Past sessions nobody marked are treated as missed: their work is still owed.
    assumed_missed = [e.id for e in open_sessions if e.start < now]

    kept: list[Session] = []
    dropped: list[str] = []
    planned: defaultdict[str, int] = defaultdict(int)
    for ev in open_sessions:
        if ev.start < now:
            continue
        task = tasks.get(ev.task_id or "")
        session = Session(task_id=ev.task_id or "", start=ev.start, end=ev.end)
        still_needed = task is not None and planned[task.id] + session.minutes <= (task.remaining_minutes or 0)
        if keep_future and still_needed:
            kept.append(session)
            planned[task.id] += session.minutes
        else:
            dropped.append(ev.id)

    shortfall = []
    for task in tasks.values():
        if task.remaining_minutes is None:
            shortfall.append(task)  # the scheduler reports it as missing an estimate
            continue
        need = task.remaining_minutes - planned[task.id]
        if need > 0:
            shortfall.append(task.model_copy(update={"estimate_minutes": need, "done_minutes": 0}))

    kept_busy = [
        Event(id=f"kept-{i}", title="kept session", start=s.start, end=s.end, source=EventSource.AGENT)
        for i, s in enumerate(kept)
    ]
    new = schedule(
        shortfall, fixed + kept_busy, prefs, start, end,
        not_before=now, committed=[(tasks[s.task_id], s) for s in kept],
    )
    return ReplanResult(
        plan=Plan(sessions=sorted(kept + new.sessions, key=lambda s: s.start), unscheduled=new.unscheduled),
        kept=kept,
        added=new.sessions,
        dropped=dropped,
        assumed_missed=assumed_missed,
    )
