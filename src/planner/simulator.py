"""Local calendar and task simulator backed by SQLite.

Stands in for real calendar and task services so the agent can be built and
evaluated without connecting personal accounts. A real connector (for example
Google Calendar) would implement the same methods.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import Event, EventSource, Plan, Preferences, Session, Task

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, data TEXT NOT NULL, start_utc TEXT NOT NULL, end_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_UPDATABLE_TASK_FIELDS = {"estimate_minutes", "done_minutes", "deadline", "priority", "status"}


def _utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def session_event_id(session: Session) -> str:
    """Deterministic id, so applying the same plan twice creates no duplicates."""
    return f"agent-{session.task_id}-{session.start.astimezone(timezone.utc):%Y%m%dT%H%M}"


class PlannerStore:
    def __init__(self, path: str | Path = ":memory:"):
        # The web UI reuses one store across its worker threads; it is single-user.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(_SCHEMA)

    def clear(self) -> None:
        with self._db:
            self._db.executescript("DELETE FROM events; DELETE FROM tasks; DELETE FROM settings;")

    @classmethod
    def from_seed(cls, seed_path: str | Path, db_path: str | Path = ":memory:") -> PlannerStore:
        store = cls(db_path)
        store.load_seed(json.loads(Path(seed_path).read_text(encoding="utf-8")))
        return store

    def load_seed(self, seed: dict) -> None:
        self.set_preferences(Preferences.model_validate(seed.get("preferences", {})))
        for raw in seed.get("events", []):
            self._put_event(Event.model_validate(raw))
        for raw in seed.get("tasks", []):
            self.put_task(Task.model_validate(raw))

    # preferences

    def get_preferences(self) -> Preferences:
        row = self._db.execute("SELECT value FROM settings WHERE key = 'preferences'").fetchone()
        return Preferences.model_validate_json(row[0]) if row else Preferences()

    def set_preferences(self, prefs: Preferences) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('preferences', ?)",
                (prefs.model_dump_json(),),
            )

    # events

    def list_events(self, start: datetime | None = None, end: datetime | None = None) -> list[Event]:
        query, params = "SELECT data FROM events", []
        conds = []
        if start is not None:
            conds.append("end_utc > ?")
            params.append(_utc(start))
        if end is not None:
            conds.append("start_utc < ?")
            params.append(_utc(end))
        if conds:
            query += " WHERE " + " AND ".join(conds)
        rows = self._db.execute(query + " ORDER BY start_utc", params).fetchall()
        return [Event.model_validate_json(r[0]) for r in rows]

    def get_event(self, event_id: str) -> Event | None:
        row = self._db.execute("SELECT data FROM events WHERE id = ?", (event_id,)).fetchone()
        return Event.model_validate_json(row[0]) if row else None

    def create_event(self, event: Event) -> bool:
        """Create an agent event. Returns False if it already exists."""
        if event.source != EventSource.AGENT:
            raise PermissionError("the agent may only create agent events")
        return self._put_event(event, replace=False)

    def delete_event(self, event_id: str) -> bool:
        event = self.get_event(event_id)
        if event is None:
            return False
        if event.source == EventSource.USER:
            raise PermissionError(f"refusing to delete user event {event_id!r}")
        with self._db:
            self._db.execute("DELETE FROM events WHERE id = ?", (event_id,))
        return True

    def mark_session(self, event_id: str, *, done: bool, minutes: int | None = None) -> Task:
        """Record the outcome of a work session.

        done: the event is kept as completed and the task's done_minutes grows
        (by `minutes`, default the session length). missed: the event is removed
        so its time goes back into the task's remaining work.
        """
        event = self.get_event(event_id)
        if event is None:
            raise KeyError(event_id)
        if event.source != EventSource.AGENT or event.task_id is None:
            raise PermissionError("only agent work sessions can be marked")
        if event.completed:
            raise ValueError(f"session {event_id!r} is already marked done")
        task = self.get_task(event.task_id)
        if task is None:
            raise KeyError(event.task_id)
        if not done:
            self.delete_event(event_id)
            return task
        worked = minutes if minutes is not None else int((event.end - event.start).total_seconds() // 60)
        self._put_event(event.model_copy(update={"completed": True}))
        return self.update_task(task.id, done_minutes=task.done_minutes + worked)

    def _put_event(self, event: Event, replace: bool = True) -> bool:
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        with self._db:
            cur = self._db.execute(
                f"{verb} INTO events (id, data, start_utc, end_utc) VALUES (?, ?, ?, ?)",
                (event.id, event.model_dump_json(), _utc(event.start), _utc(event.end)),
            )
        return cur.rowcount == 1

    # tasks

    def list_tasks(self, include_done: bool = False) -> list[Task]:
        tasks = [Task.model_validate_json(r[0]) for r in self._db.execute("SELECT data FROM tasks ORDER BY id")]
        return tasks if include_done else [t for t in tasks if t.status == "todo"]

    def get_task(self, task_id: str) -> Task | None:
        row = self._db.execute("SELECT data FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return Task.model_validate_json(row[0]) if row else None

    def put_task(self, task: Task) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO tasks (id, data) VALUES (?, ?)", (task.id, task.model_dump_json())
            )

    def update_task(self, task_id: str, **changes) -> Task:
        unknown = set(changes) - _UPDATABLE_TASK_FIELDS
        if unknown:
            raise ValueError(f"cannot update task fields: {sorted(unknown)}")
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        updated = Task.model_validate({**task.model_dump(), **changes})
        self.put_task(updated)
        return updated

    # execution

    def apply_plan(self, plan: Plan) -> list[Event]:
        """Write approved sessions as agent events. Idempotent."""
        created = []
        for session in plan.sessions:
            task = self.get_task(session.task_id)
            event = Event(
                id=session_event_id(session),
                title=task.title if task else session.task_id,
                start=session.start,
                end=session.end,
                source=EventSource.AGENT,
                task_id=session.task_id,
            )
            if self.create_event(event):
                created.append(event)
        return created
