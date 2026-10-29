from datetime import date, datetime

from planner.intake import ParsedRequest, TaskDraft, drafts_to_tasks, parse_request
from planner.llm import CachedLLM, ScriptedLLM
from planner.models import Preferences, TaskKind

PREFS = Preferences(timezone="Europe/London")
PARSED = ParsedRequest(
    tasks=[
        TaskDraft(title="Study for exam", kind="deep", estimate_minutes=480, estimate_is_guess=True,
                  session_minutes=None, deadline="2026-11-06 09:00", priority=1),
        TaskDraft(title="Workout", kind="workout", estimate_minutes=180, estimate_is_guess=False,
                  session_minutes=60, deadline=None, priority=2),
    ],
    constraints=["do not move existing appointments"],
    questions=["Is 8 hours of study enough for the exam?"],
)


def test_parse_request_includes_calendar_context():
    llm = ScriptedLLM({ParsedRequest: PARSED})
    parse_request(llm, "Plan my week", prefs=PREFS, start=date(2026, 11, 2), end=date(2026, 11, 8),
                  today=date(2026, 11, 2))
    assert "Friday = 2026-11-06" in llm.prompts[0]


def test_drafts_to_tasks():
    exam, workout = drafts_to_tasks(PARSED, PREFS)
    assert exam.deadline == datetime(2026, 11, 6, 9, tzinfo=PREFS.tz)
    assert workout.kind == TaskKind.WORKOUT
    assert (workout.min_session_minutes, workout.max_session_minutes, workout.max_sessions_per_day) == (60, 60, 1)


def test_cache_avoids_repeat_calls(tmp_path):
    inner = ScriptedLLM({ParsedRequest: PARSED})
    llm = CachedLLM(inner, cache_dir=tmp_path)
    for _ in range(2):
        assert llm.generate(system="s", prompt="p", schema=ParsedRequest) == PARSED
    assert len(inner.calls) == 1
    assert [c.cached for c in llm.calls] == [False, True]


def test_cache_reports_the_model_that_answered(tmp_path):
    inner = ScriptedLLM({ParsedRequest: PARSED})
    inner.model = "primary"
    original_generate = inner.generate

    def answered_by_fallback(**kwargs):
        result = original_generate(**kwargs)
        inner.calls[-1].model = "fallback"
        return result

    inner.generate = answered_by_fallback
    llm = CachedLLM(inner, cache_dir=tmp_path)
    for _ in range(2):
        llm.generate(system="s", prompt="p", schema=ParsedRequest)
    assert [c.model for c in llm.calls] == ["fallback", "fallback"]
