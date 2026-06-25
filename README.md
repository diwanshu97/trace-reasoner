# Trace-Reasoner — build

Multi-agent distributed-trace anomaly localization. This directory is the
*build* (code), kept separate from the checkpoint writeups in `../`.

Per the locked schedule, Week 1 is **data + eval harness before any agent code**.
This increment delivers that foundation:

```
trace_reasoner/
  trace.py            canonical Span / Trace model — what every tool walks
  datasets/
    base.py           Dataset / Example / GroundTruth contract
    synthetic.py      span trees with a KNOWN injected anomaly (free labels)
    nezha.py          loader for the Nezha dataset (data/Nezha/)
  eval/
    metrics.py        RCA accuracy, localization F1, Brier + ECE / accuracy-coverage / escalation (CP6)
    harness.py        run any localizer over a Dataset and score it
  baselines/
    slowest_leaf.py   naive "blame the slowest leaf" localizer (the punching bag)
  agent/              Condition A: the monolithic ReAct loop (react.py)
  multiagent/         Condition B: specialist analysts + synthesizer (LangGraph);
                      Condition C: Tree-of-Thought beam search (beam.py)
  rag/                precedent retrieval — FAISS over BGE-small (Checkpoint 3)
  safety/             Checkpoint 6 control system: guardrails, trust/risk router, safety metrics
  mcp/                MCP server exposing the tools to an external client
tests/                pytest: model, synthetic labels, metrics, harness, agents, safety
eval_baseline.py      run the baseline end-to-end and print a report
eval_conditions.py    A/B/C iso-budget comparison offline (+ --safe for CP6); one table
run_agent.py          live smoke test of Condition A on Claude
run_multiagent.py     live smoke test of Condition B on Claude
run_tot.py            live smoke test of Condition C on Claude
run_local.py          run A/B/C on a local open model (Ollama) — the free live backend
app.py                Streamlit demo UI: pick a trace, condition, and brain (mock/local/Claude)
```

A **localizer** is any `Callable[[Trace], Prediction]`. Each ablation condition
is just a more sophisticated localizer that plugs into the same harness, so the
comparison is measured on one ruler:

- **A — monolithic ReAct** (`agent/react.py`): one agent, one context, all tools.
- **B — multi-agent** (`multiagent/`): three prompt-scoped specialist analysts
  (latency, dependency, pattern) fan out over the trace in isolated contexts and
  write structured findings into a shared belief state; a deterministic synthesizer
  reconciles them into a ranked, calibrated prediction with a capped, targeted
  re-dispatch. Coordination is a LangGraph state machine; the synthesizer stays
  plain Python so orchestration adds no hidden LLM calls to the iso-token budget.
- **C — +Tree-of-Thought** (`multiagent/beam.py`): the B specialists become thought
  *generators*; a separate critic scores each candidate span on the four Checkpoint-4
  criteria (anomaly grounding, critical-path coverage, precedent, verification) over the
  same grounded tools; a plain-Python beam controller (width 3, depth ≤ 4) prunes and
  expands. A branch is hard-pruned only by a negative tool result, never a weak early
  score, so the subtle true cause survives. The tool budget is the controlled variable
  (beam width × depth), keeping the iso-token A/B/C comparison honest.

Every condition talks to its brain through one `LLMClient` protocol, so the same agent
loop runs on any of **three backends** with no code change:

- **mock** (`agent/mock.py`) — deterministic heuristics; no key, no network. Powers the tests
  and the demo UI; the reproducible default.
- **Claude** (`agent/anthropic_client.py`, `claude-opus-4-8` by default) — the live runs.
- **local open model** (`agent/local_client.py`) — Qwen2.5-7B / Llama-3.1-8B served by Ollama
  (or any OpenAI-compatible server). The *free* live backend, and a model-capability axis: do the
  A/B/C architecture gains survive when the model is ~10× smaller than Claude? Run `run_local.py`.

So the whole suite is green offline on the mock, and the same comparison can be re-run on Claude
or a laptop-local model on the identical iso-budget ruler.

Evaluation is graded on calibration over raw accuracy (the Checkpoint 1 commitment):
beyond top-k RCA accuracy and F1, the harness reports **ECE**, the **accuracy–coverage**
curve, and the **escalation rate** (CP6) — the instruments behind "inconclusive within budget".

**Safety (Checkpoint 6).** `safety/` wraps any condition in a control system without touching its
task logic. `SafeLocalizer(inner, baseline)` is itself a `Localizer`, so it scores on the same
harness: input **guardrails** (size cap, prompt-injection scan, secret redaction) run before the
agent; **source verification** drops hallucinated spans and enforces the output schema after; a
**trust/risk router** then assigns a lane — `AUTO` (confident, in-distribution, grounded),
`REVIEW` (delivered but flagged), or `ESCALATE` (abstain → on-call SRE). An ESCALATE empties the
prediction, so it lands on the same accuracy–coverage curve as the agent's own "inconclusive"
path. `safety.evaluation` adds router metrics: escalation precision, autonomy rate, lane mix.

## Setup

```bash
cd capstone/trace-reasoner
uv venv .venv && uv pip install -p .venv pytest
.venv/bin/python -m pytest          # all green, offline
.venv/bin/python eval_baseline.py   # baseline numbers on synthetic traces
```

Condition B needs LangGraph; live runs need the Anthropic SDK and a key:

```bash
.venv/bin/pip install -e '.[agent,multiagent,rag]'
ANTHROPIC_API_KEY=sk-... .venv/bin/python run_multiagent.py
```

The A/B/C comparison and the demo UI run offline — no key:

```bash
.venv/bin/python eval_conditions.py            # one table; add --safe for the CP6 control system
uv pip install -p .venv -e '.[multiagent,ui]'  # streamlit + langgraph
.venv/bin/python -m streamlit run app.py       # pick a trace + condition, see the routing decision
```

Free live backend — a local open model via Ollama (no API key, runs on a laptop):

```bash
brew install ollama && ollama serve            # in another terminal
ollama pull qwen2.5:7b-instruct                # ~4.7 GB; or llama3.1:8b-instruct-q4_0
.venv/bin/python run_local.py --n 3            # A/B/C on the local model (slow; keep n small)
```

## Datasets

- **synthetic** — offline, deterministic, ground-truth for free. Used for tests
  and fast iteration.
- **Nezha** — primary labeled dataset (FSE'23, IntelligentDDS/Nezha). Clone into
  `data/Nezha/`:
  ```bash
  git clone --depth 1 https://github.com/IntelligentDDS/Nezha.git data/Nezha
  ```
