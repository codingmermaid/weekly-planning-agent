from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from planner.runtime import open_store, planning_window
from planner.scheduler import schedule
from planner.simulator import PlannerStore

APP = Path(__file__).resolve().parent.parent / "app.py"
SEED = APP.parent / "data" / "seed_week.json"


def test_app_replan_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANNER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "ollama")  # no network: the client only connects when called

    # A week that was planned and approved on Monday morning.
    store = open_store()
    for task in PlannerStore.from_seed(SEED).list_tasks():
        store.put_task(task)
    store.apply_plan(schedule(store.list_tasks(), store.list_events(), store.get_preferences(), *planning_window()))

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Week Planner"

    at.date_input(key="now_date").set_value(date(2026, 11, 4)).run()
    at.button(key="replan").click().run()
    assert not at.exception
    assert at.subheader[0].value == "Proposed replan"
    assert any("treated as missed" in i.value for i in at.info)

    approve = next(b for b in at.button if b.label == "Approve and add to calendar")
    approve.click().run()
    assert not at.exception
    assert at.success and at.success[0].value.startswith("Done")
