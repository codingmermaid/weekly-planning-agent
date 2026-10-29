"""Deterministic greedy scheduler.

The LLM never does calendar arithmetic. It proposes tasks and estimates; this
module turns them into concrete sessions, and the validator checks the result.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Literal, Sequence

from .availability import Interval, day_windows, free_slots
from .models import Event, Plan, Preferences, Session, Task, TaskKind, TaskStatus, Unscheduled

Strategy = Literal["balanced", "earliest"]


def split_minutes(total: int, max_session: int) -> list[int]:
    """Split work into the fewest near-equal sessions no longer than max_session."""
    if total <= 0:
        return []
    n = math.ceil(total / max_session)
    base, extra = divmod(total, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


class _DailyLimits:
    """Tracks per-day load so placement respects preferences and task limits."""

    def __init__(self, prefs: Preferences):
        self.prefs = prefs
        self.deep_minutes: defaultdict[date, int] = defaultdict(int)
        self.workout_days: set[date] = set()
        self.task_sessions: Counter[tuple[str, date]] = Counter()

    def allows(self, task: Task, day: date, minutes: int) -> bool:
        if self.task_sessions[(task.id, day)] >= task.max_sessions_per_day:
            return False
        if task.kind == TaskKind.DEEP and self.deep_minutes[day] + minutes > self.prefs.max_deep_minutes_per_day:
            return False
        if task.kind == TaskKind.WORKOUT and self.prefs.one_workout_per_day and day in self.workout_days:
            return False
        return True

    def add(self, task: Task, day: date, minutes: int) -> None:
        self.task_sessions[(task.id, day)] += 1
        if task.kind == TaskKind.DEEP:
            self.deep_minutes[day] += minutes
        if task.kind == TaskKind.WORKOUT:
            self.workout_days.add(day)


def _urgency(task: Task) -> tuple:
    # Earliest deadline first, then priority, then biggest task first.
    deadline = task.deadline.timestamp() if task.deadline else math.inf
    return (deadline, task.priority, -(task.remaining_minutes or 0))


def busy_minutes_by_day(events: list[Event], prefs: Preferences, start: date, end: date) -> defaultdict[date, int]:
    """Minutes of existing appointments inside each day's working window."""
    load: defaultdict[date, int] = defaultdict(int)
    for w_start, w_end in day_windows(prefs, start, end):
        day = w_start.date()
        for ev in events:
            overlap = min(ev.end, w_end) - max(ev.start, w_start)
            if overlap > timedelta(0):
                load[day] += int(overlap.total_seconds() // 60)
    return load


def _place(
    task: Task,
    minutes: int,
    slots: list[Interval],
    limits: _DailyLimits,
    load: defaultdict[date, int],
    prefs: Preferences,
    strategy: Strategy,
) -> tuple[Session | None, bool]:
    """Returns the placed session (or None) and whether daily limits blocked a slot that fit."""
    need = timedelta(minutes=minutes)
    pad = timedelta(minutes=prefs.buffer_minutes)
    best: tuple[tuple, int, date] | None = None
    limited = False
    for i, (start, end) in enumerate(slots):
        finish = start + need
        if task.deadline and finish > task.deadline:
            break  # slots are chronological, so no later slot can work either
        if finish > end:
            continue
        day = start.astimezone(prefs.tz).date()
        if not limits.allows(task, day, minutes):
            limited = True
            continue
        # "earliest" takes the first fit; "balanced" takes the least-loaded day, earliest slot on ties.
        key = (start,) if strategy == "earliest" else (load[day], start)
        if best is None or key < best[0]:
            best = (key, i, day)
        if strategy == "earliest":
            break
    if best is None:
        return None, limited

    _, i, day = best
    start, end = slots[i]
    finish = start + need
    limits.add(task, day, minutes)
    load[day] += minutes
    if finish + pad < end:
        slots[i] = (finish + pad, end)
    else:
        del slots[i]
    return Session(task_id=task.id, start=start, end=finish), False


def schedule(
    tasks: list[Task],
    events: list[Event],
    prefs: Preferences,
    start: date,
    end: date,
    *,
    not_before: datetime | None = None,
    strategy: Strategy = "balanced",
    committed: Sequence[tuple[Task, Session]] = (),
) -> Plan:
    """Place open tasks into free time between start and end (inclusive dates).

    "balanced" spreads work across the least-busy days (counting existing
    appointments); "earliest" packs work as early as possible and is kept as a
    baseline for evaluation.

    committed: sessions that already exist and stay put (used by replanning).
    They must also be included in `events` so their time counts as busy.
    """
    slots = free_slots(prefs, events, start, end)
    if not_before is not None:
        slots = [(max(s, not_before), e) for s, e in slots if e > not_before]

    limits = _DailyLimits(prefs)
    for task, session in committed:
        limits.add(task, session.start.astimezone(prefs.tz).date(), session.minutes)
    load = busy_minutes_by_day(events, prefs, start, end)
    plan = Plan()
    for task in sorted((t for t in tasks if t.status == TaskStatus.TODO), key=_urgency):
        remaining = task.remaining_minutes
        if remaining is None:
            plan.unscheduled.append(Unscheduled(task_id=task.id, minutes=0, reason="missing time estimate"))
            continue
        chunks = split_minutes(remaining, task.max_session_minutes)
        for i, minutes in enumerate(chunks):
            session, limited = _place(task, minutes, slots, limits, load, prefs, strategy)
            if session is None:
                cause = "daily limits reached" if limited else "no free slot"
                reason = f"{cause} before deadline" if task.deadline else f"{cause} in planning window"
                plan.unscheduled.append(Unscheduled(task_id=task.id, minutes=sum(chunks[i:]), reason=reason))
                break
            plan.sessions.append(session)

    plan.sessions.sort(key=lambda s: s.start)
    return plan
