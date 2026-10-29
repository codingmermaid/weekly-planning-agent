# Weekly Planning Agent

A single AI agent that plans your week around existing appointments. Tell it what you need to
fit in ("I have an exam on Friday, two client deadlines and three workouts"), answer a few
questions, approve the plan, and it writes work sessions to your calendar. When sessions get
done or missed, it replans the rest of the week without moving what's already settled.

Built as a practice project for agentic AI: one agent coordinating several tools through a
graph, with human approval before anything changes. It runs entirely on free tools.

![Streamlit UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b) ![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1c3c3c) ![Gemini free tier](https://img.shields.io/badge/LLM-Gemini%20free%20tier-4285f4)

## What it does

- **Reads** your calendar and tasks (a local simulator, so no personal accounts are needed).
- **Asks** about missing information: time estimates, deadlines, session lengths.
- **Plans** by splitting big tasks into work sessions, spread across the least busy days.
- **Checks** every plan against hard rules: no overlaps, working hours only, deadlines met,
  existing appointments untouched, daily limits respected.
- **Waits for approval** before writing anything. You can approve, cancel, or ask for changes.
- **Replans** after you mark sessions done, partly done or missed, moving only the work you still owe.
- **Says so honestly** when something can't fit, instead of quietly dropping it or breaking a rule.

## How it works

The key design choice: **the LLM handles language, and plain code handles the calendar.**

| Part | Job | LLM? |
|---|---|---|
| Intake | Turn your request into tasks and clarifying questions | Yes (Gemini) |
| Scheduler | Find free time and place sessions | No |
| Validator | Check every plan against the hard rules | No |
| Graph | Control the flow: ask, plan, check, approve, write | No |

Because the scheduler and validator are deterministic, a weaker or misbehaving model can make a
poor estimate, but it can never produce an invalid calendar.

### Agent graph

```
plan mode:    intake -> (clarify -> intake)* -> schedule -> validate
replan mode:  replan -> validate
then:         validate -> (repair -> validate)* -> approve -> execute
                                                         \-> revise -> intake
                                                         \-> cancelled
```

- **clarify** and **approve** are interrupts: the graph pauses until you answer.
- Run state is checkpointed to SQLite, so you can quit at a question and resume later.
- **repair** removes any session that breaks a rule and reports that time as unscheduled.

### Replanning

- **Fixed:** your appointments and completed sessions. Never moved.
- **Kept:** future sessions that are still needed stay exactly where they are.
- **Rescheduled:** only the missing work, placed after "now".
- **Assumed missed:** past sessions you never marked. Their work is still owed.

Replanning uses no LLM, so it's instant and free.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On macOS or Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

### Add a free Gemini API key

1. Create a key at https://aistudio.google.com/apikey. No billing or card is needed.
2. Copy `.env.example` to `.env` and paste the key after `GEMINI_API_KEY=` (no quotes).

`.env` is ignored by git. Never commit it or paste the key anywhere else. Don't enable billing on
the key's project, so the free tier stays a hard cap.

### Run the web app

```bash
.venv\Scripts\streamlit.exe run app.py
```

- **Plan:** describe your week, answer the questions, then approve, cancel or request changes.
- **Calendar:** colour-coded week view.
- **Sessions:** mark sessions done or missed, then replan from a simulated "now".

The sample week (2–8 November 2026) is in the future, so pick a simulated date and time in the
sidebar to try marking sessions and replanning.

### Or use the terminal

```bash
.venv\Scripts\python.exe scripts\agent.py "I have an exam on Friday, two client deadlines and three workouts."
.venv\Scripts\python.exe scripts\agent.py --thread <id>        # resume a paused run
.venv\Scripts\python.exe scripts\agent.py --sessions           # list sessions and their ids
.venv\Scripts\python.exe scripts\agent.py --done <id> --minutes 60
.venv\Scripts\python.exe scripts\agent.py --missed <id>
.venv\Scripts\python.exe scripts\agent.py --replan --now "2026-11-04 09:00"   # add --full to move future sessions too
.venv\Scripts\python.exe scripts\agent.py --reset              # restore the sample week
```

`scripts\demo.py` plans the sample week without the full agent, and works offline.

## Models and the free tier

| Setting (`.env`) | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Or `ollama` for a local model |
| `GEMINI_MODEL` | `gemini-3.5-flash` | About 2 seconds per call |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.8-flash,gemini-3.1-flash-lite` | Tried in order when a model is busy or slow |
| `OLLAMA_MODEL` | `qwen3:8b` | Offline and private, no limits |

- Free-tier models get overloaded at times. The client retries briefly, then falls back to the
  next model.
- Responses are cached in `.cache/`, so re-running the same request costs nothing.
- Each call's model, latency and token counts are recorded, and shown in the app's sidebar.
- Model names change. If you get "model not found", pick a current one from
  https://ai.google.dev/gemini-api/docs/models.
- Free-tier prompts may be used by Google to improve its products, so stick to the simulated
  calendar rather than real personal data.

## Project structure

| Path | Role |
|---|---|
| `app.py` | Streamlit web app |
| `scripts/agent.py` | Interactive terminal agent |
| `scripts/demo.py` | Offline demo of the scheduler |
| `src/planner/models.py` | Events, tasks, preferences, plans |
| `src/planner/simulator.py` | SQLite calendar and task store |
| `src/planner/availability.py` | Working hours minus existing events |
| `src/planner/scheduler.py` | Balanced placement (or `earliest` as a baseline) |
| `src/planner/validator.py` | Hard-rule checks every plan must pass |
| `src/planner/intake.py` | LLM step: request to tasks and questions |
| `src/planner/llm.py` | Gemini and Ollama clients, retries, fallback, cache |
| `src/planner/graph.py` | LangGraph workflow with interrupts |
| `src/planner/replan.py` | Minimal-change replanning |
| `src/planner/render.py` | Text and HTML week views |
| `src/planner/runtime.py` | Shared setup for the app and CLI |
| `data/seed_week.json` | Sample week: appointments, preferences, demo tasks |
| `tests/` | Unit and flow tests |

## Tests

```bash
.venv\Scripts\python.exe -m pytest -q
```

The tests use a scripted stand-in for the LLM, so they run offline and use no quota. They cover
the scheduler, validator, simulator, every graph path, replanning, the Gemini retry and fallback
logic, and a headless run of the web app.

## Ideas for later

- Let feedback change preferences, such as "no work on Sunday".
- Smarter placement: slack before deadlines, focused work in the mornings.
- Expose the calendar tools as an MCP server, and move the planning procedure into a skill file.
- A small evaluation: a handful of sample weeks, compared against asking the model to schedule in
  one prompt.
- Optional Google Calendar connection behind a dry-run flag.
