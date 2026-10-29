# Weekly Planning Agent

A single agent that plans a week around existing appointments. The LLM handles
language (parsing requests, estimates, clarifying questions). Deterministic code
handles the calendar (free time, placement, constraint checks).

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\demo.py
```

To use Gemini, copy `.env.example` to `.env` and add a free key from
https://aistudio.google.com/apikey. Then:

```bash
.venv\Scripts\python.exe scripts\demo.py --request "I have an exam on Friday, two client deadlines and three workouts. Don't move existing appointments."
```

Web UI (plan, answer questions, approve, mark sessions, replan):

```bash
.venv\Scripts\streamlit.exe run app.py
```

Or run the full agent in the terminal (asks questions, waits for approval, writes to the simulated calendar):

```bash
.venv\Scripts\python.exe scripts\agent.py "I have an exam on Friday, two client deadlines and three workouts."
.venv\Scripts\python.exe scripts\agent.py --thread <id>   # resume a paused run
.venv\Scripts\python.exe scripts\agent.py --reset         # restore the sample week
```

Record what happened and replan the rest of the week:

```bash
.venv\Scripts\python.exe scripts\agent.py --sessions                  # list sessions and ids
.venv\Scripts\python.exe scripts\agent.py --done <id> --minutes 60    # partly done
.venv\Scripts\python.exe scripts\agent.py --missed <id>
.venv\Scripts\python.exe scripts\agent.py --replan --now "2026-11-04 09:00"   # add --full to move future sessions too
```

Replanning keeps appointments and completed sessions fixed, leaves still-needed future sessions
where they are, and schedules only the missing work after `now`. Past sessions never marked are
treated as missed.

## Agent graph

```
plan mode:   intake -> (clarify -> intake)* -> schedule -> validate
replan mode: replan -> validate
then:        validate -> (repair -> validate)* -> approve -> execute | (revise -> intake) | cancelled
```

`clarify` and `approve` are interrupts. Run state is checkpointed to SQLite, so a run can pause
and resume across processes. Nothing is written to the calendar before approval.

## Layout

| File | Role |
|---|---|
| `src/planner/models.py` | Events, tasks, preferences, plans |
| `src/planner/simulator.py` | SQLite calendar and task store (stands in for real services) |
| `src/planner/availability.py` | Working hours minus existing events |
| `src/planner/scheduler.py` | Greedy placement (earliest deadline first) |
| `src/planner/validator.py` | Hard-constraint checks every plan must pass |
| `src/planner/llm.py` | Gemini / Ollama clients, structured output, disk cache, usage records |
| `src/planner/intake.py` | LLM step: free text to task drafts and clarifying questions |
| `src/planner/graph.py` | LangGraph workflow with clarify/approve interrupts |
| `src/planner/replan.py` | Minimal-change replanning |
| `src/planner/render.py` | Text rendering of a week plan |
| `scripts/agent.py` | Interactive agent CLI |
| `app.py` | Streamlit web UI |
| `src/planner/runtime.py` | Shared data locations and setup for the CLI and UI |
| `data/seed_week.json` | Sample week |

## Next steps

1. Let feedback change preferences (e.g. "no work on Sunday"), not just tasks.
2. Smarter placement: slack before deadlines, deep work in mornings; optionally OR-Tools CP-SAT.
3. MCP server over the simulator, and a weekly-planning skill file.
4. Scenario suite and evaluation: hard-constraint pass rate, soft score, replan stability, cost and latency vs. a single-prompt baseline.
