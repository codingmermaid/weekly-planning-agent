from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from planner.graph import build_graph, new_run, pending_interrupts
from planner.intake import ParsedRequest, TaskDraft
from planner.llm import ScriptedLLM
from planner.models import EventSource, Plan, Session
from planner.scheduler import schedule
from planner.validator import check_user_events_unchanged

EXAM = TaskDraft(title="Study for exam", kind="deep", estimate_minutes=240, estimate_is_guess=False,
                 session_minutes=None, deadline="2026-11-06 09:00", priority=1)
WORKOUT = TaskDraft(title="Workout", kind="workout", estimate_minutes=120, estimate_is_guess=False,
                    session_minutes=60, deadline=None, priority=2)
CLEAR = ParsedRequest(tasks=[EXAM, WORKOUT], constraints=[], questions=[])
UNCLEAR = ParsedRequest(tasks=[EXAM.model_copy(update={"estimate_minutes": None})], constraints=[],
                        questions=["How many hours do you need to study?"])


def start(store, window, llm, **kwargs):
    graph = build_graph(store, llm, checkpointer=InMemorySaver(), **kwargs)
    state = new_run("Plan my week", *window, run_id="t1")
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(state, config)
    return graph, config


def agent_events(store):
    return [e for e in store.list_events() if e.source == EventSource.AGENT]


def test_approved_plan_is_written_without_touching_appointments(store, window):
    before = store.list_events()
    graph, config = start(store, window, ScriptedLLM({ParsedRequest: CLEAR}))

    [payload] = pending_interrupts(graph, config)
    assert payload["kind"] == "approval"
    assert agent_events(store) == []  # nothing written before approval

    graph.invoke(Command(resume={"approved": True}), config)
    final = graph.get_state(config).values
    assert final["status"] == "done"
    assert len(agent_events(store)) == len(final["plan"]["sessions"]) == len(final["created_event_ids"])
    assert check_user_events_unchanged(before, store.list_events()) == []


def test_asks_clarifying_questions_then_uses_the_answer(store, window):
    llm = ScriptedLLM({ParsedRequest: [UNCLEAR, CLEAR]})
    graph, config = start(store, window, llm)

    [payload] = pending_interrupts(graph, config)
    assert payload["kind"] == "clarify"
    assert payload["questions"] == ["How many hours do you need to study?"]

    graph.invoke(Command(resume="About 4 hours"), config)
    assert "A: About 4 hours" in llm.prompts[-1]
    assert pending_interrupts(graph, config)[0]["kind"] == "approval"


def test_stops_asking_after_max_rounds(store, window):
    graph, config = start(store, window, ScriptedLLM({ParsedRequest: UNCLEAR}))
    for _ in range(2):
        graph.invoke(Command(resume="not sure"), config)
    [payload] = pending_interrupts(graph, config)
    assert payload["kind"] == "approval"
    # Still no estimate, so the task is reported, not guessed.
    assert payload["plan"]["unscheduled"][0]["reason"] == "missing time estimate"


def test_rejection_with_feedback_replans(store, window):
    llm = ScriptedLLM({ParsedRequest: CLEAR})
    graph, config = start(store, window, llm)
    graph.invoke(Command(resume={"approved": False, "feedback": "Only 3 hours of study"}), config)

    assert "- Only 3 hours of study" in llm.prompts[-1]
    assert pending_interrupts(graph, config)[0]["kind"] == "approval"
    assert agent_events(store) == []


def test_rejection_without_feedback_cancels(store, window):
    graph, config = start(store, window, ScriptedLLM({ParsedRequest: CLEAR}))
    graph.invoke(Command(resume={"approved": False}), config)
    assert graph.get_state(config).values["status"] == "cancelled"
    assert agent_events(store) == []


def test_validator_catches_a_bad_planner_and_repair_removes_it(store, window):
    def overlapping_scheduler(tasks, events, prefs, start_date, end_date):
        plan = schedule(tasks, events, prefs, start_date, end_date)
        dentist = next(e for e in events if e.title == "Dentist")
        workout = next(t for t in tasks if t.kind == "workout")
        plan.sessions.append(Session(task_id=workout.id, start=dentist.start, end=dentist.end))
        return plan

    graph, config = start(store, window, ScriptedLLM({ParsedRequest: CLEAR}), scheduler=overlapping_scheduler)
    state = graph.get_state(config).values
    assert any(entry.startswith("repair:") for entry in state["log"])

    [payload] = pending_interrupts(graph, config)
    plan = Plan.model_validate(payload["plan"])
    assert all(s.task_id != "t1-2" for s in plan.sessions)  # every workout session dropped
    assert any("removed by validator" in u.reason for u in plan.unscheduled)
