from datetime import date, datetime

from planner.models import Event, Plan, Preferences, Session, Task
from planner.validator import check_user_events_unchanged, validate_plan

PREFS = Preferences(timezone="Europe/London")
MON = date(2026, 11, 2)
TUE = date(2026, 11, 3)
TASKS = [
    Task(id="study", title="Study", estimate_minutes=240, deadline=datetime(2026, 11, 3, 12, tzinfo=PREFS.tz)),
    Task(id="gym", title="Workout", kind="workout", estimate_minutes=120,
         min_session_minutes=60, max_session_minutes=60, max_sessions_per_day=1),
]
EVENTS = [Event(id="e", title="Dentist", start=datetime(2026, 11, 2, 14, tzinfo=PREFS.tz),
                end=datetime(2026, 11, 2, 15, tzinfo=PREFS.tz))]


def s(task_id: str, day: date, start: tuple, end: tuple) -> Session:
    return Session(
        task_id=task_id,
        start=datetime(day.year, day.month, day.day, *start, tzinfo=PREFS.tz),
        end=datetime(day.year, day.month, day.day, *end, tzinfo=PREFS.tz),
    )


def codes(*sessions: Session) -> set[str]:
    return {v.code for v in validate_plan(Plan(sessions=list(sessions)), TASKS, EVENTS, PREFS)}


def test_valid_plan_has_no_violations():
    assert codes(s("study", MON, (9,), (11,)), s("gym", MON, (16,), (17,))) == set()


def test_detects_each_hard_constraint():
    assert codes(s("study", MON, (13, 30), (14, 30))) == {"OVERLAP_EVENT"}
    assert codes(s("study", MON, (7,), (8,))) == {"OUTSIDE_HOURS"}
    assert codes(s("study", TUE, (11,), (13,))) == {"AFTER_DEADLINE"}
    assert codes(s("study", MON, (9,), (11,)), s("gym", MON, (10,), (11,))) == {"OVERLAP_SESSION"}
    assert codes(s("gym", MON, (9,), (10,)), s("gym", MON, (16,), (17,))) == {"WORKOUT_SAME_DAY", "TOO_MANY_SESSIONS"}
    assert codes(s("study", MON, (9,), (13,)), s("study", MON, (15,), (17,))) == {"OVER_ALLOCATED"}
    assert codes(s("ghost", MON, (9,), (10,))) == {"UNKNOWN_TASK"}


def test_detects_changed_user_events():
    moved = EVENTS[0].model_copy(update={"start": datetime(2026, 11, 2, 13, tzinfo=PREFS.tz)})
    assert [v.code for v in check_user_events_unchanged(EVENTS, [moved])] == ["USER_EVENT_MODIFIED"]
    assert [v.code for v in check_user_events_unchanged(EVENTS, [])] == ["USER_EVENT_DELETED"]
