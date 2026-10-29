"""Core domain models shared by the simulator, scheduler, validator and agent."""
from __future__ import annotations

from datetime import datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator


class TaskKind(str, Enum):
    DEEP = "deep"        # focused work: studying, client deliverables
    WORKOUT = "workout"
    ADMIN = "admin"      # light work: email, errands


class EventSource(str, Enum):
    USER = "user"    # existing appointments: the agent must never move or delete these
    AGENT = "agent"  # work sessions the agent created: safe to replan


class TaskStatus(str, Enum):
    TODO = "todo"
    DONE = "done"


def _require_aware(*values: datetime | None) -> None:
    for v in values:
        if v is not None and v.tzinfo is None:
            raise ValueError("datetimes must be timezone-aware")


class Event(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    source: EventSource = EventSource.USER
    task_id: str | None = None
    completed: bool = False  # agent sessions the user marked done; never replanned

    @model_validator(mode="after")
    def _check_times(self) -> Event:
        _require_aware(self.start, self.end)
        if self.end <= self.start:
            raise ValueError(f"event {self.id!r} ends before it starts")
        return self


class Task(BaseModel):
    id: str
    title: str
    kind: TaskKind = TaskKind.DEEP
    estimate_minutes: int | None = Field(None, ge=0)  # None means unknown: ask the user
    done_minutes: int = Field(0, ge=0)
    deadline: datetime | None = None
    priority: int = Field(2, ge=1, le=3)  # 1 is highest
    min_session_minutes: int = Field(30, gt=0)
    max_session_minutes: int = Field(120, gt=0)
    max_sessions_per_day: int = Field(2, gt=0)
    status: TaskStatus = TaskStatus.TODO

    @model_validator(mode="after")
    def _check(self) -> Task:
        _require_aware(self.deadline)
        if self.min_session_minutes > self.max_session_minutes:
            raise ValueError(f"task {self.id!r}: min session is longer than max session")
        return self

    @property
    def remaining_minutes(self) -> int | None:
        if self.estimate_minutes is None:
            return None
        return max(self.estimate_minutes - self.done_minutes, 0)


class Preferences(BaseModel):
    timezone: str = "UTC"
    day_start: time = time(9, 0)
    day_end: time = time(21, 0)
    buffer_minutes: int = Field(10, ge=0)
    max_deep_minutes_per_day: int = Field(360, gt=0)
    workdays: list[int] = Field(default_factory=lambda: list(range(7)))  # 0 = Monday
    one_workout_per_day: bool = True

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


class Session(BaseModel):
    """A proposed block of work on one task. Becomes an agent Event once approved."""
    task_id: str
    start: datetime
    end: datetime

    @property
    def minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)


class Unscheduled(BaseModel):
    task_id: str
    minutes: int
    reason: str


class Plan(BaseModel):
    sessions: list[Session] = Field(default_factory=list)
    unscheduled: list[Unscheduled] = Field(default_factory=list)
