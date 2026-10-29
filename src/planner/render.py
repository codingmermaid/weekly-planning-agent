"""Rendering of a week plan: plain text for the CLI, an HTML grid for the web UI."""
from __future__ import annotations

import html
from collections import defaultdict
from datetime import date, datetime, timedelta

from .models import Event, EventSource, Plan, Preferences, Task

# kind -> (colour, legend label). Colours read on light and dark backgrounds.
BLOCK_STYLES = {
    "cal": ("#6b7280", "Appointment"),
    "done": ("#16a34a", "Done"),
    "planned": ("#2563eb", "Planned"),
    "new": ("#d97706", "Proposed"),
}
Block = tuple[datetime, datetime, str, str]  # start, end, kind, title


def event_blocks(events: list[Event]) -> list[Block]:
    blocks = []
    for ev in events:
        kind = "cal" if ev.source == EventSource.USER else "done" if ev.completed else "planned"
        blocks.append((ev.start, ev.end, kind, ev.title))
    return blocks


def week_html(blocks: list[Block], prefs: Preferences, start: date, end: date, hour_px: int = 34) -> str:
    """A week grid: one column per day, blocks positioned by time within working hours."""
    tz = prefs.tz
    day_start = prefs.day_start.hour * 60 + prefs.day_start.minute
    day_end = prefs.day_end.hour * 60 + prefs.day_end.minute
    height = (day_end - day_start) * hour_px / 60
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    by_day = defaultdict(list)
    for b_start, b_end, kind, title in blocks:
        by_day[b_start.astimezone(tz).date()].append((b_start.astimezone(tz), b_end.astimezone(tz), kind, title))

    hour_lines = "".join(
        f'<div class="wk-hr" style="top:{(h * 60 - day_start) * hour_px / 60}px"></div>'
        for h in range(day_start // 60 + 1, day_end // 60 + 1)
        if h * 60 < day_end
    )
    labels = "".join(
        f'<span style="top:{(h * 60 - day_start) * hour_px / 60}px">{h:02d}:00</span>'
        for h in range(-(-day_start // 60), day_end // 60 + 1)
    )

    cells = ['<div></div>'] + [f'<div class="wk-hd">{d:%a}<br><small>{d:%d %b}</small></div>' for d in days]
    cells.append(f'<div class="wk-lbl" style="height:{height}px">{labels}</div>')
    for d in days:
        parts = [hour_lines]
        for b_start, b_end, kind, title in sorted(by_day[d]):
            top_min = max(b_start.hour * 60 + b_start.minute, day_start) - day_start
            bottom_min = min(b_end.hour * 60 + b_end.minute, day_end) - day_start
            if bottom_min <= top_min:
                continue  # outside working hours
            colour = BLOCK_STYLES[kind][0]
            tip = html.escape(f"{b_start:%H:%M}-{b_end:%H:%M} {title}")
            parts.append(
                f'<div class="wk-blk" title="{tip}" style="top:{top_min * hour_px / 60}px;'
                f'height:{(bottom_min - top_min) * hour_px / 60 - 2}px;background:{colour}">'
                f'<b>{b_start:%H:%M}</b> {html.escape(title)}</div>'
            )
        cells.append(f'<div class="wk-col" style="height:{height}px">{"".join(parts)}</div>')

    legend = "".join(
        f'<span><i style="background:{colour}"></i>{label}</span>' for colour, label in BLOCK_STYLES.values()
    )
    style = (
        "<style>"
        ".wk-wrap{overflow-x:auto;padding-bottom:10px}"
        f".wk{{display:grid;grid-template-columns:44px repeat({len(days)},minmax(96px,1fr));gap:4px;"
        "font-size:12px;min-width:740px}"
        ".wk-hd{text-align:center;font-weight:600;padding-bottom:4px;line-height:1.2}"
        ".wk-hd small{font-weight:400;opacity:.7}"
        ".wk-col{position:relative;border-radius:6px;background:rgba(128,128,128,.08)}"
        ".wk-hr{position:absolute;left:0;right:0;border-top:1px dashed rgba(128,128,128,.25)}"
        ".wk-lbl{position:relative}"
        ".wk-lbl span{position:absolute;right:4px;transform:translateY(-50%);font-size:11px;opacity:.6}"
        ".wk-blk{position:absolute;left:3px;right:3px;border-radius:5px;padding:2px 5px;color:#fff;"
        "overflow:hidden;line-height:1.25;box-sizing:border-box}"
        ".wk-legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:12px}"
        ".wk-legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}"
        "</style>"
    )
    return (
        f'{style}<div class="wk-wrap"><div class="wk">{"".join(cells)}</div></div>'
        f'<div class="wk-legend">{legend}</div>'
    )


def format_plan(plan: Plan, tasks: list[Task], events: list[Event], prefs: Preferences) -> str:
    titles = {t.id: t.title for t in tasks}
    by_day = defaultdict(list)
    for ev in events:
        tag = "done" if ev.completed else "cal "
        by_day[ev.start.astimezone(prefs.tz).date()].append((ev.start, ev.end, tag, ev.title))
    for s in plan.sessions:
        by_day[s.start.astimezone(prefs.tz).date()].append((s.start, s.end, "PLAN", titles.get(s.task_id, s.task_id)))

    lines = []
    for day in sorted(by_day):
        booked = sum((end - start).total_seconds() for start, end, _, _ in by_day[day]) / 3600
        lines.append(f"\n{day:%A %d %b}  ({booked:.1f}h booked)")
        for start, end, tag, title in sorted(by_day[day]):
            lines.append(f"  {start.astimezone(prefs.tz):%H:%M}-{end.astimezone(prefs.tz):%H:%M}  [{tag}] {title}")
    if plan.unscheduled:
        lines.append("\nNot scheduled:")
        for u in plan.unscheduled:
            amount = f"{u.minutes} min" if u.minutes else "all"
            lines.append(f"  - {titles.get(u.task_id, u.task_id)}: {amount} ({u.reason})")
    return "\n".join(lines)
