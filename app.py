"""Web UI for the weekly planning agent.

    .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime, time

import streamlit as st
from langgraph.types import Command

from planner.graph import build_graph, new_replan_run, new_run, pending_interrupts
from planner.llm import make_llm
from planner.models import EventSource, Plan, Task
from planner.render import event_blocks, week_html
from planner.replan import is_fixed
from planner.runtime import open_checkpointer, open_store, planning_window, reset_store
from planner.simulator import session_event_id

EXAMPLE_REQUEST = (
    "I have an exam on Friday, two client deadlines and three workouts to fit in. "
    "Plan my week without moving existing appointments."
)

st.set_page_config(page_title="Week Planner", page_icon="🗓️", layout="wide")


@st.cache_resource
def resources():
    return open_store(), open_checkpointer()


@st.cache_resource
def get_llm():
    return make_llm()


store, checkpointer = resources()
try:
    llm, llm_error = get_llm(), None
except Exception as exc:  # missing key or bad provider: replanning still works without an LLM
    llm, llm_error = None, str(exc)
graph = build_graph(store, llm, checkpointer=checkpointer)
prefs = store.get_preferences()
start, end = planning_window()

ss = st.session_state
ss.setdefault("thread", None)
ss.setdefault("result", None)
ss.setdefault("error", None)


def config() -> dict:
    return {"configurable": {"thread_id": ss.thread}}


def pending() -> list[dict]:
    return pending_interrupts(graph, config()) if ss.thread else []


def advance(payload, message: str) -> None:
    """Start a run (payload is a new state) or resume one (payload is the user's answer)."""
    try:
        with st.spinner(message):
            if isinstance(payload, dict) and "run_id" in payload:
                ss.thread = payload["run_id"]
                graph.invoke(payload, config())
            else:
                graph.invoke(Command(resume=payload), config())
    except Exception as exc:  # e.g. Gemini unavailable: keep the run so it can be retried
        ss.error = str(exc)
        return
    if not pending():
        ss.result = graph.get_state(config()).values
        ss.thread = None


def simulated_now() -> datetime:
    return datetime.combine(ss.now_date, ss.now_time, prefs.tz)


# sidebar

with st.sidebar:
    st.header("Settings")
    st.date_input("Simulated date", value=start, min_value=start, max_value=end, key="now_date")
    st.time_input("Simulated time", value=time(9, 0), key="now_time", step=1800)
    st.caption("The sample week is in the future. Pick a 'now' inside it to try marking sessions and replanning.")

    st.divider()
    if llm:
        st.write(f"**LLM:** {llm.provider} / {llm.model}")
        calls = llm.calls
        if calls:
            tokens_in = sum(c.input_tokens or 0 for c in calls)
            tokens_out = sum(c.output_tokens or 0 for c in calls)
            cached = sum(c.cached for c in calls)
            st.caption(f"{len(calls)} calls ({cached} cached), {tokens_in} in / {tokens_out} out tokens this session")
    else:
        st.warning(f"No LLM available, so planning is off. Replanning still works.\n\n{llm_error}")

    st.divider()
    if st.button("Reset to sample week", disabled=bool(ss.thread)):
        reset_store(store)
        ss.result = None
        st.toast("Calendar restored to the sample week.")


st.title("Week Planner")

if ss.error:
    st.error(ss.error)
    ss.error = None


# the run waiting on the user, if any

waiting = pending()
if waiting:
    payload = waiting[0]
    with st.container(border=True):
        if payload["kind"] == "clarify":
            st.subheader("A few questions first")
            for q in payload["questions"]:
                st.markdown(f"- {q}")
            answer = st.text_area("Your answer", placeholder="Answer all the questions in one message.")
            if st.button("Send answers", type="primary", disabled=not answer.strip()):
                advance(answer.strip(), "Updating the tasks...")
                st.rerun()
        else:
            replanning = payload.get("mode") == "replan"
            st.subheader("Proposed replan" if replanning else "Proposed plan")
            plan = Plan.model_validate(payload["plan"])
            tasks = [Task.model_validate(t) for t in payload["tasks"]]
            titles = {t.id: t.title for t in tasks}

            events = store.list_events()
            if replanning:
                events = [e for e in events if is_fixed(e)]  # the plan already holds every open session
            existing = {e.id for e in store.list_events()}
            blocks = event_blocks(events) + [
                (s.start, s.end, "planned" if session_event_id(s) in existing else "new", titles.get(s.task_id, s.task_id))
                for s in plan.sessions
            ]
            st.html(week_html(blocks, prefs, start, end))

            if replanning:
                r = payload["replan"]
                st.info(
                    f"Keeps {r['kept']} sessions, adds {r['added']}, removes {len(r['dropped'])} no longer needed. "
                    f"{len(r['assumed_missed'])} past sessions were never marked and are treated as missed."
                )
            for u in plan.unscheduled:
                amount = f"{u.minutes} min" if u.minutes else "All of it"
                st.warning(f"**{titles.get(u.task_id, u.task_id)}**: {amount} not scheduled ({u.reason})")
            if not replanning:
                with st.expander("Tasks the agent understood"):
                    st.dataframe(
                        [{"Task": t.title, "Kind": t.kind.value, "Minutes": t.estimate_minutes,
                          "Deadline": t.deadline.astimezone(prefs.tz).strftime("%a %d %b %H:%M") if t.deadline else ""}
                         for t in tasks],
                        hide_index=True,
                    )

            approve_col, cancel_col = st.columns(2)
            if approve_col.button("Approve and add to calendar", type="primary", use_container_width=True):
                advance({"approved": True}, "Writing to the calendar...")
                st.rerun()
            if cancel_col.button("Cancel", use_container_width=True):
                advance({"approved": False}, "Cancelling...")
                st.rerun()
            if not replanning:
                feedback = st.text_input("Or describe changes", placeholder="e.g. the exam needs 10 hours, not 8")
                if st.button("Request changes", disabled=not feedback.strip()):
                    advance({"approved": False, "feedback": feedback.strip()}, "Replanning...")
                    st.rerun()

elif ss.result:
    result = ss.result
    status = result.get("status")
    if status == "done":
        added, removed = len(result.get("created_event_ids", [])), len(result.get("removed_event_ids", []))
        st.success(f"Done: added {added} and removed {removed} events." if removed
                   else f"Done: added {added} events to the calendar.")
    elif status == "cancelled":
        st.info("Cancelled. Nothing was changed.")
    else:
        st.error(f"Run ended with status {status!r}.")
        for v in result.get("violations") or []:
            st.write(f"- {v['code']}: {v['message']}")
    with st.expander("Agent trace"):
        st.code("\n".join(result.get("log", [])), language=None)
    if st.button("Dismiss"):
        ss.result = None
        st.rerun()


# main tabs

plan_tab, calendar_tab, sessions_tab = st.tabs(["Plan", "Calendar", "Sessions"])

with plan_tab:
    request = st.text_area("What do you need to fit into the week?", value=EXAMPLE_REQUEST, height=100)
    busy = bool(waiting)
    if st.button("Plan my week", type="primary", disabled=busy or llm is None or not request.strip()):
        advance(new_run(request.strip(), start, end), "Reading your request...")
        st.rerun()
    if busy:
        st.caption("Finish the run above first.")

with calendar_tab:
    st.html(week_html(event_blocks(store.list_events()), prefs, start, end))

with sessions_tab:
    now = simulated_now()
    sessions = [e for e in store.list_events() if e.source == EventSource.AGENT]
    if not sessions:
        st.write("No work sessions yet. Plan your week first.")
    for ev in sessions:
        length = int((ev.end - ev.start).total_seconds() // 60)
        when, minutes_col, done_col, missed_col = st.columns([3, 1.2, 1, 1.3], vertical_alignment="center")
        past = "" if ev.start >= now else " · past"
        when.markdown(f"**{ev.start.astimezone(prefs.tz):%a %H:%M}** {ev.title}  \n"
                      f"<small>{length} min{past}</small>", unsafe_allow_html=True)
        if ev.completed:
            done_col.markdown("✅ Done")
            continue
        worked = minutes_col.number_input("Minutes", min_value=0, value=length, step=15,
                                          key=f"min-{ev.id}", label_visibility="collapsed")
        if done_col.button("Done", key=f"done-{ev.id}", disabled=busy):
            store.mark_session(ev.id, done=True, minutes=int(worked))
            st.rerun()
        if missed_col.button("Missed", key=f"missed-{ev.id}", disabled=busy):
            store.mark_session(ev.id, done=False)
            st.rerun()

    st.divider()
    full = st.checkbox("Also move future sessions (full replan)")
    if st.button(f"Replan from {now:%a %H:%M}", type="primary", disabled=busy or not sessions, key="replan"):
        advance(new_replan_run(start, end, now, keep_future=not full), "Replanning...")
        st.rerun()
