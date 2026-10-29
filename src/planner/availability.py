"""Free-time computation: working-hour windows minus existing events."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from .models import Event, Preferences

Interval = tuple[datetime, datetime]


def day_windows(prefs: Preferences, start: date, end: date) -> list[Interval]:
    """Working-hour windows for each allowed day in [start, end], inclusive."""
    tz = prefs.tz
    windows = []
    d = start
    while d <= end:
        if d.weekday() in prefs.workdays:
            windows.append((datetime.combine(d, prefs.day_start, tz), datetime.combine(d, prefs.day_end, tz)))
        d += timedelta(days=1)
    return windows


def subtract(free: list[Interval], busy: list[Interval]) -> list[Interval]:
    """Remove busy intervals from free intervals. Keeps chronological order."""
    busy = sorted(busy)
    result = []
    for start, end in free:
        cursor = start
        for b_start, b_end in busy:
            if b_end <= cursor or b_start >= end:
                continue
            if b_start > cursor:
                result.append((cursor, b_start))
            cursor = max(cursor, b_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def free_slots(prefs: Preferences, events: list[Event], start: date, end: date) -> list[Interval]:
    pad = timedelta(minutes=prefs.buffer_minutes)
    busy = [(ev.start - pad, ev.end + pad) for ev in events]
    return subtract(day_windows(prefs, start, end), busy)
