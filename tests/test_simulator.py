import pytest

from planner.scheduler import schedule
from planner.validator import check_user_events_unchanged


def test_apply_plan_is_idempotent_and_preserves_user_events(store, window):
    before = store.list_events()
    plan = schedule(store.list_tasks(), before, store.get_preferences(), *window)

    created = store.apply_plan(plan)
    assert len(created) == len(plan.sessions)
    assert store.apply_plan(plan) == []  # second apply creates nothing
    assert check_user_events_unchanged(before, store.list_events()) == []


def test_agent_cannot_delete_user_events(store):
    with pytest.raises(PermissionError):
        store.delete_event("ev-dentist")


def test_agent_can_delete_its_own_events(store, window):
    plan = schedule(store.list_tasks(), store.list_events(), store.get_preferences(), *window)
    event = store.apply_plan(plan)[0]
    assert store.delete_event(event.id) is True


def test_update_task_rejects_unknown_fields(store):
    assert store.update_task("exam", done_minutes=120).remaining_minutes == 360
    with pytest.raises(ValueError):
        store.update_task("exam", title="renamed")
