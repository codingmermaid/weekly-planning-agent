"""LLM step: turn a free-text planning request into structured task drafts.

The schemas here are what the model fills in, so they stay flat and use plain
strings for dates. drafts_to_tasks converts them into validated Task models.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from .llm import LLMClient
from .models import Preferences, Task, TaskKind


class TaskDraft(BaseModel):
    title: str = Field(description="Short task name, e.g. 'Study for exam'")
    kind: Literal["deep", "workout", "admin"] = Field(
        description="deep = focused work (study, client work); workout = exercise; admin = light tasks"
    )
    estimate_minutes: int | None = Field(
        description="Total minutes of work needed. null if not stated and you cannot reasonably guess."
    )
    estimate_is_guess: bool = Field(description="true if you guessed the estimate instead of the user stating it")
    session_minutes: int | None = Field(
        description="Fixed length of each session when implied (e.g. 60 for 'one-hour workouts'), else null"
    )
    deadline: str | None = Field(description="Local deadline as 'YYYY-MM-DD HH:MM', or null if none")
    priority: int = Field(description="1 = highest, 2 = normal, 3 = lowest")


class ParsedRequest(BaseModel):
    tasks: list[TaskDraft]
    constraints: list[str] = Field(description="Constraints the user stated, e.g. 'do not move existing appointments'")
    questions: list[str] = Field(
        description="Short clarifying questions for missing or guessed information. Empty if nothing is missing."
    )


SYSTEM_PROMPT = """You turn a weekly planning request into structured tasks. You do not schedule anything.

Rules:
- Resolve relative dates ("Friday", "tomorrow") using the calendar given.
- A deadline on a day with no time means the start of that working day ({day_start}).
- If the user states how long something takes, use it. Otherwise guess a reasonable estimate,
  set estimate_is_guess to true, and add a question asking the user to confirm it.
- If you cannot reasonably guess, set estimate_minutes to null and ask.
- Repeated activities ("three workouts") are one task: estimate_minutes = count x session length and
  session_minutes = session length. If the length is not stated, assume 60 and ask.
- Separate deliverables are separate tasks, even when unnamed: "two client deadlines" means two tasks
  ("Client deadline 1", "Client deadline 2"), each with its own deadline question if the date is unknown.
- Never invent tasks the user did not mention.
"""


def _calendar_context(prefs: Preferences, start: date, end: date, today: date) -> str:
    days = []
    d = start
    while d <= end:
        days.append(f"{d:%A} = {d:%Y-%m-%d}")
        d += timedelta(days=1)
    return (
        f"Today is {today:%A %Y-%m-%d}. Timezone: {prefs.timezone}.\n"
        f"Planning window: {start} to {end}.\nDays: " + "; ".join(days)
    )


def parse_request(
    llm: LLMClient, text: str, *, prefs: Preferences, start: date, end: date, today: date
) -> ParsedRequest:
    system = SYSTEM_PROMPT.format(day_start=prefs.day_start.strftime("%H:%M"))
    prompt = f"{_calendar_context(prefs, start, end, today)}\n\nRequest:\n{text}"
    return llm.generate(system=system, prompt=prompt, schema=ParsedRequest)


def drafts_to_tasks(parsed: ParsedRequest, prefs: Preferences, id_prefix: str = "req") -> list[Task]:
    tasks = []
    for i, draft in enumerate(parsed.tasks, start=1):
        deadline = None
        if draft.deadline:
            deadline = datetime.strptime(draft.deadline, "%Y-%m-%d %H:%M").replace(tzinfo=prefs.tz)
        fields = {
            "id": f"{id_prefix}-{i}",
            "title": draft.title,
            "kind": TaskKind(draft.kind),
            "estimate_minutes": draft.estimate_minutes,
            "deadline": deadline,
            "priority": min(max(draft.priority, 1), 3),
        }
        if draft.session_minutes:
            fields["min_session_minutes"] = fields["max_session_minutes"] = draft.session_minutes
        if draft.kind == "workout":
            fields["max_sessions_per_day"] = 1
        tasks.append(Task(**fields))
    return tasks
