from datetime import date, datetime, timedelta

from planner.availability import free_slots
from planner.models import Event, Preferences, Task
from planner.scheduler import schedule, split_minutes
from planner.validator import validate_plan

PREFS = Preferences(timezone="Europe/London", buffer_minutes=10)
MON = date(2026, 11, 2)


def at(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=PREFS.tz)


def test_split_minutes():
    assert split_minutes(300, 120) == [100, 100, 100]
    assert split_minutes(180, 60) == [60, 60, 60]
    assert split_minutes(45, 120) == [45]
    assert split_minutes(0, 120) == []


def test_free_slots_subtract_events_with_buffer():
    events = [Event(id="e", title="Meeting", start=at(MON, 10), end=at(MON, 11))]
    assert free_slots(PREFS, events, MON, MON) == [
        (at(MON, 9), at(MON, 9, 50)),
        (at(MON, 11, 10), at(MON, 21)),
    ]


def test_seed_week_plan_is_valid_and_complete(store, window):
    tasks, events, prefs = store.list_tasks(), store.list_events(), store.get_preferences()
    plan = schedule(tasks, events, prefs, *window)

    assert validate_plan(plan, tasks, events, prefs) == []
    # Everything with an estimate fits; the task without one is flagged, not guessed.
    assert [(u.task_id, u.reason) for u in plan.unscheduled] == [("inbox", "missing time estimate")]
    scheduled = {t.id: 0 for t in tasks}
    for s in plan.sessions:
        scheduled[s.task_id] += s.minutes
    for t in tasks:
        if t.estimate_minutes is not None:
            assert scheduled[t.id] == t.remaining_minutes


def test_infeasible_task_reports_unscheduled_minutes():
    task = Task(id="big", title="Huge report", estimate_minutes=600, deadline=at(MON, 13))
    plan = schedule([task], [], PREFS, MON, MON + timedelta(days=6))

    assert validate_plan(plan, [task], [], PREFS) == []
    # 600 min splits into 5 x 120. Only 09:00-11:00 fits: after the 10 min buffer, 11:10-13:00 is too short.
    assert sum(s.minutes for s in plan.sessions) == 120
    assert plan.unscheduled[0].minutes == 480
    assert "deadline" in plan.unscheduled[0].reason


def test_reason_names_daily_limits_when_they_block():
    # Free time exists on Monday, but the 360 min deep-work cap stops a third 2h session.
    task = Task(id="t", title="Report", estimate_minutes=480, max_sessions_per_day=4, deadline=at(MON, 21))
    plan = schedule([task], [], PREFS, MON, MON)
    assert sum(s.minutes for s in plan.sessions) == 360
    assert plan.unscheduled[0].reason == "daily limits reached before deadline"


def test_not_before_skips_past_time():
    task = Task(id="t", title="Task", estimate_minutes=60)
    plan = schedule([task], [], PREFS, MON, MON, not_before=at(MON, 15, 30))
    assert plan.sessions[0].start == at(MON, 15, 30)


def test_balanced_spreads_work_across_days():
    task = Task(id="t", title="Essay", estimate_minutes=360, max_session_minutes=120)
    week = (MON, MON + timedelta(days=6))
    balanced = schedule([task], [], PREFS, *week)
    earliest = schedule([task], [], PREFS, *week, strategy="earliest")
    assert len({s.start.date() for s in balanced.sessions}) == 3
    assert len({s.start.date() for s in earliest.sessions}) == 2  # 2 sessions/day cap, packed early


def test_balanced_avoids_days_with_appointments():
    tue = MON + timedelta(days=1)
    busy_monday = [Event(id="e", title="Workshop", start=at(MON, 9), end=at(MON, 17))]
    task = Task(id="t", title="Report", estimate_minutes=60)
    plan = schedule([task], busy_monday, PREFS, MON, tue)
    assert plan.sessions[0].start.date() == tue


def test_one_workout_per_day():
    task = Task(id="w", title="Workout", kind="workout", estimate_minutes=180,
                min_session_minutes=60, max_session_minutes=60, max_sessions_per_day=3)
    plan = schedule([task], [], PREFS, MON, MON + timedelta(days=6))
    days = [s.start.date() for s in plan.sessions]
    assert len(days) == 3 and len(set(days)) == 3
