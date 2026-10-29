"""Shared setup for the CLI and the web UI: data locations, store and checkpoints.

Set PLANNER_DATA_DIR to keep the calendar and run state somewhere other than data/.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .simulator import PlannerStore

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "data" / "seed_week.json"


def data_dir() -> Path:
    return Path(os.getenv("PLANNER_DATA_DIR") or ROOT / "data")


def store_path() -> Path:
    return data_dir() / "planner.db"


def checkpoint_path() -> Path:
    return data_dir() / "checkpoints.db"


def _seed() -> dict:
    return json.loads(SEED.read_text(encoding="utf-8"))


def planning_window() -> tuple[date, date]:
    window = _seed()["window"]
    return date.fromisoformat(window["start"]), date.fromisoformat(window["end"])


def reset_store(store: PlannerStore) -> None:
    """Restore the sample week: appointments and preferences only.

    The seed's task list is for the offline demo; the agent's tasks come from requests.
    """
    store.clear()
    store.load_seed({k: v for k, v in _seed().items() if k != "tasks"})


def open_store() -> PlannerStore:
    path = store_path()
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    store = PlannerStore(path)
    if is_new:
        reset_store(store)
    return store


def open_checkpointer() -> SqliteSaver:
    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
