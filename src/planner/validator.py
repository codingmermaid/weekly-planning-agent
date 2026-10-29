"""Hard-constraint checks for any proposed plan, whoever produced it.

Every plan (from the scheduler, a repair step, or an LLM) must pass
validate_plan before it is shown to the user or written to the calendar.
This module deliberately re-derives everything instead of trusting the
scheduler's bookkeeping.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel

from .models import Event, EventSource, Plan, Preferences, Task, TaskKind


class Violation(BaseModel):
    code: str
    message: str
    task_id: str | None = None


def validate_plan(plan: Plan, tasks: list[Task], events: list[Event], prefs: Preferences) -> list[Violation]:
    tz = prefs.tz
    by_id = {t.id: t for t in tasks}
    out: list[Violation] = []

    def add(code: str, message: str, task_id: str | None = None) -> None:
        out.append(Violation(code=code, message=message, task_id=task_id))

    valid = []
    for s in plan.sessions:
        task = by_id.get(s.task_id)
        if task is None:
            add("UNKNOWN_TASK", f"session refers to unknown task {s.task_id!r}", s.task_id)
            continue
        if s.start.tzinfo is None or s.end.tzinfo is None:
            add("NAIVE_DATETIME", f"{task.title}: session times must be timezone-aware", task.id)
            continue
        if s.end <= s.start:
            add("EMPTY_SESSION", f"{task.title}: session ends before it starts", task.id)
            continue
        valid.append(s)

        label = f"{task.title} {s.start.astimezone(tz):%a %H:%M}"
        local_start, local_end = s.start.astimezone(tz), s.end.astimezone(tz)
        if (
            local_start.date() != local_end.date()
            or local_start.weekday() not in prefs.workdays
            or local_start.time() < prefs.day_start
            or local_end.time() > prefs.day_end
        ):
            add("OUTSIDE_HOURS", f"{label} is outside allowed hours", task.id)
        if task.deadline and s.end > task.deadline:
            add("AFTER_DEADLINE", f"{label} ends after the deadline", task.id)
        for ev in events:
            if s.start < ev.end and ev.start < s.end:
                add("OVERLAP_EVENT", f"{label} overlaps '{ev.title}'", task.id)

    latest = None
    for s in sorted(valid, key=lambda s: s.start):
        if latest is not None and s.start < latest.end:
            add("OVERLAP_SESSION", f"sessions for {latest.task_id!r} and {s.task_id!r} overlap", s.task_id)
        if latest is None or s.end > latest.end:
            latest = s

    minutes: defaultdict[str, int] = defaultdict(int)
    task_day_sessions: Counter = Counter()
    deep_by_day: defaultdict = defaultdict(int)
    workouts_by_day: Counter = Counter()
    for s in valid:
        task = by_id[s.task_id]
        day = s.start.astimezone(tz).date()
        minutes[task.id] += s.minutes
        task_day_sessions[(task.id, day)] += 1
        if task.kind == TaskKind.DEEP:
            deep_by_day[day] += s.minutes
        if task.kind == TaskKind.WORKOUT:
            workouts_by_day[day] += 1

    for task_id, total in minutes.items():
        remaining = by_id[task_id].remaining_minutes
        if remaining is None:
            add("MISSING_ESTIMATE", f"{by_id[task_id].title} was scheduled without an estimate", task_id)
        elif total > remaining:
            add("OVER_ALLOCATED", f"{by_id[task_id].title}: {total} min scheduled, {remaining} needed", task_id)
    for (task_id, day), n in task_day_sessions.items():
        if n > by_id[task_id].max_sessions_per_day:
            add("TOO_MANY_SESSIONS", f"{by_id[task_id].title}: {n} sessions on {day}", task_id)
    for day, total in deep_by_day.items():
        if total > prefs.max_deep_minutes_per_day:
            add("DEEP_CAP", f"{total} min of deep work on {day} exceeds {prefs.max_deep_minutes_per_day}")
    if prefs.one_workout_per_day:
        for day, n in workouts_by_day.items():
            if n > 1:
                add("WORKOUT_SAME_DAY", f"{n} workouts on {day}")
    return out


def check_user_events_unchanged(before: list[Event], after: list[Event]) -> list[Violation]:
    """Existing appointments must survive execution exactly as they were."""
    after_by_id = {e.id: e for e in after}
    out = []
    for ev in before:
        if ev.source != EventSource.USER:
            continue
        now = after_by_id.get(ev.id)
        if now is None:
            out.append(Violation(code="USER_EVENT_DELETED", message=f"'{ev.title}' was deleted"))
        elif now != ev:
            out.append(Violation(code="USER_EVENT_MODIFIED", message=f"'{ev.title}' was changed"))
    return out
