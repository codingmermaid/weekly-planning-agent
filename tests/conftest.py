import json
from datetime import date
from pathlib import Path

import pytest

from planner.simulator import PlannerStore

SEED = Path(__file__).resolve().parent.parent / "data" / "seed_week.json"


@pytest.fixture
def store() -> PlannerStore:
    return PlannerStore.from_seed(SEED)


@pytest.fixture
def window() -> tuple[date, date]:
    raw = json.loads(SEED.read_text(encoding="utf-8"))["window"]
    return date.fromisoformat(raw["start"]), date.fromisoformat(raw["end"])
