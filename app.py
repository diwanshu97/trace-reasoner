"""Trace-Reasoner demo UI (Checkpoint 6 / capstone presentation).

A Streamlit front end over the build: pick a synthetic trace, a condition (A/B/C), and — the point
of the project — a **brain** (the LLM backend). The same agent orchestration runs on a deterministic
mock, a local open model (Qwen2.5-7B via Ollama), or Claude, so you can watch better orchestration
(A -> B -> C) recover good results even from a weaker model. It then shows the trace, the ranked
root-cause hypotheses with calibrated confidence, and the AUTO / REVIEW / ESCALATE routing decision.

It calls the same engine as the CLI (trace_reasoner.eval.compare + safety.SafeLocalizer), so the
demo and the experiment never disagree.

    .venv/bin/python -m streamlit run app.py

Mock needs nothing. Local needs `ollama serve` + `ollama pull qwen2.5:7b-instruct`. Claude needs
ANTHROPIC_API_KEY. The mock is instant; live brains are gated behind a button (local ~1 min/trace).
"""

from __future__ import annotations

import os
import urllib.request

import pandas as pd
import streamlit as st

from trace_reasoner.datasets.synthetic import SyntheticDataset, generate_example, normal_traces
from trace_reasoner.eval.compare import CONDITION_LABELS, build_conditions, compare
from trace_reasoner.safety.router import Lane
from trace_reasoner.tools.baseline import LatencyBaseline
from trace_reasoner.tools.walk_tree import survey

st.set_page_config(page_title="Trace-Reasoner", page_icon="🔭", layout="wide")

_LANE_STYLE = {
    Lane.AUTO: ("✅ AUTO — deliver autonomously", "#1a7f37", "#dafbe1"),
    Lane.REVIEW: ("⚠️ REVIEW — deliver, flagged for a human", "#9a6700", "#fff8c5"),
    Lane.ESCALATE: ("⛔ ESCALATE — abstain, route to on-call SRE", "#cf222e", "#ffebe9"),
}

_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_LOCAL_MODEL = "qwen2.5:7b-instruct"
_DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"


# --- backends (the brain) -----------------------------------------------------
@st.cache_resource
def get_baseline() -> LatencyBaseline:
    """The fault-free latency baseline, built once and reused across reruns."""
    return LatencyBaseline.from_traces(normal_traces(100, seed=0))


def make_llm(brain: str, model: str):
    """Construct the injected LLMClient for a brain, or None for the offline mock."""
    if brain == "local":
        from trace_reasoner.agent.local_client import LocalClient

        return LocalClient(model=model)
    if brain == "claude":
        from trace_reasoner.agent.anthropic_client import AnthropicClient

        return AnthropicClient(model=model)
    return None  # mock: build_conditions falls back to the deterministic heuristics


def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(_OLLAMA_URL + "/api/version", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def localize(brain: str, model: str, cond_key: str, safe: bool, trace):
    """Run one trace through one condition on the chosen brain. May call a live LLM (slow).

    Returns a RoutedDecision when safe=True, else a Prediction — matching the CLI engine.
    """
    conds = build_conditions(baseline=get_baseline(), live_llm=make_llm(brain, model), safe=safe)
    loc = conds.localizers[cond_key]
    return loc.decide(trace) if safe else loc(trace)


# --- rendering ----------------------------------------------------------------
def lane_banner(lane: Lane) -> None:
    label, fg, bg = _LANE_STYLE[lane]
    st.markdown(
        f"<div style='padding:0.6rem 1rem;border-radius:8px;background:{bg};"
        f"color:{fg};font-weight:600;font-size:1.05rem'>{label}</div>",
        unsafe_allow_html=True,
    )


def trace_table(trace, suspects: set[str]) -> pd.DataFrame:
    """One row per span, sorted by self-time, with the hottest suspects flagged."""
    rows = []
    for s in sorted(trace.spans, key=lambda s: trace.self_time_ms(s.span_id), reverse=True):
        rows.append({
            "": "🔥" if s.span_id in suspects else "",
            "span_id": s.span_id,
            "service": s.service,
            "operation": s.operation,
            "self_time_ms": round(trace.self_time_ms(s.span_id), 1),
            "duration_ms": round(s.duration_ms, 1),
            "status": s.status,
        })
    return pd.DataFrame(rows)


def render_result(safe: bool, result, truth: set[str]) -> None:
    """Render a RoutedDecision (safe) or a bare Prediction (unsafe) into the right panel."""
    if safe:
        decision = result
        lane_banner(decision.lane)
        delivered, raw = decision.prediction, decision.raw_prediction
        st.caption(
            f"OOD signal {decision.ood_score:.2f} · groundedness {decision.groundedness:.2f}"
            + (f" · dropped {len(decision.dropped_spans)} hallucinated span(s)" if decision.dropped_spans else "")
        )
        with st.expander("Why this routing decision"):
            for r in decision.reasons:
                st.write(f"• {r}")
    else:
        delivered = raw = result

    shown = delivered.ranked if delivered.ranked else raw.ranked
    st.caption("Ranked root-cause hypotheses" + ("" if delivered.ranked else " — withheld; shown for inspection"))
    if not shown:
        st.info("Inconclusive within budget — no span cleared the confidence floor.")
    for h in shown:
        hit = "🎯 " if h.span_id in truth else ""
        st.write(f"{hit}**{h.span_id}** · {h.fault_family or 'unknown'} · conf {h.confidence:.2f}")
        st.progress(min(1.0, h.confidence))
        if h.evidence:
            st.caption(h.evidence[0])


# --- sidebar: the brain selector ----------------------------------------------
def brain_selector() -> tuple[str, str, bool]:
    """Pick the LLM backend. Returns (brain_key, model, ready)."""
    st.sidebar.header("🧠 Model backend")
    st.sidebar.caption("The same A/B/C orchestration runs on any brain — that's the experiment.")
    label = st.sidebar.radio(
        "Brain",
        ["Mock — offline, instant", "Local — Ollama (free)", "Claude — API"],
        help="Mock is deterministic and instant. Local/Claude run a real LLM (slower).",
    )
    brain = {"Mock — offline, instant": "mock", "Local — Ollama (free)": "local", "Claude — API": "claude"}[label]

    model, ready = "mock", True
    if brain == "local":
        model = st.sidebar.text_input("Ollama model tag", _DEFAULT_LOCAL_MODEL)
        if ollama_up():
            st.sidebar.success("Ollama is running.")
        else:
            ready = False
            st.sidebar.error("Ollama not reachable on :11434.")
            st.sidebar.code("ollama serve\nollama pull qwen2.5:7b-instruct", language="bash")
        st.sidebar.caption("≈1 min/trace for A; B/C fan out into specialists, so several minutes.")
    elif brain == "claude":
        model = st.sidebar.text_input("Claude model", _DEFAULT_CLAUDE_MODEL)
        if os.environ.get("ANTHROPIC_API_KEY"):
            st.sidebar.success("ANTHROPIC_API_KEY is set.")
        else:
            ready = False
            st.sidebar.error("ANTHROPIC_API_KEY not set in this environment.")
        st.sidebar.caption("A few seconds per trace; costs API tokens.")
    else:
        st.sidebar.success("No key or server needed.")

    return brain, model, ready


# --- tab 1: localize one trace ------------------------------------------------
def single_trace_tab(brain: str, model: str, ready: bool) -> None:
    st.subheader("Localize one trace")

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    seed = c1.number_input("Trace seed", min_value=0, max_value=999_999, value=42, step=1)
    fault = c2.selectbox("Injected fault", ["latency", "error", "none"], index=0)
    cond_key = c3.selectbox("Condition", list(CONDITION_LABELS), format_func=lambda k: CONDITION_LABELS[k])
    safe = c4.toggle("CP6 safety system", value=True, help="Wrap the condition in guardrails + the trust/risk router")

    ex = generate_example(seed=int(seed), fault=fault)
    trace = ex.trace
    truth = set(ex.ground_truth.root_cause_span_ids)
    s = survey(trace)
    suspects = {v.span_id for v in s.hottest_by_self_time}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spans", s.n_spans)
    m2.metric("End-to-end", f"{s.duration_ms:.0f} ms")
    m3.metric("Services", len(s.services))
    m4.metric("Error spans", s.error_spans)

    left, right = st.columns([1.3, 1])
    with left:
        st.caption("Trace spans (🔥 = hottest by self-time, the first suspects)")
        st.dataframe(trace_table(trace, suspects), use_container_width=True, hide_index=True, height=360)

    with right:
        key = (brain, model, int(seed), fault, cond_key, safe)
        results = st.session_state.setdefault("results", {})

        if brain == "mock":
            result = localize(brain, model, cond_key, safe, trace)  # instant
        else:
            if st.button(f"▶ Run condition {cond_key} on {model}", disabled=not ready, type="primary"):
                with st.spinner(f"Running on {brain}… live LLM, this can take a while."):
                    try:
                        results[key] = localize(brain, model, cond_key, safe, trace)
                    except Exception as exc:  # server down mid-run, tool-call failure, etc.
                        st.error(f"Run failed: {exc}")
            result = results.get(key)
            if result is None:
                st.info(f"Press **Run** to localize this trace on {brain}. (Mock is instant if you just want to explore.)")

        if result is not None:
            render_result(safe, result, truth)
            st.divider()
            gt = ", ".join(truth) if truth else "none (fault-free trace)"
            st.caption(f"Ground truth: **{gt}** · injected fault: *{ex.ground_truth.fault_family or 'none'}* · brain: *{brain}*")


# --- tab 2: A/B/C comparison --------------------------------------------------
def comparison_tab(brain: str, model: str, ready: bool) -> None:
    st.subheader("A/B/C iso-budget comparison")
    st.caption(
        "The central experiment: three conditions at an equal tool budget, scored on one ruler. "
        "On the mock, accuracy saturates and calibration (ECE) separates them; on a weaker brain, "
        "watch whether B/C recover what A misses."
    )
    is_live = brain != "mock"
    c1, c2, c3 = st.columns(3)
    n_max = 5 if is_live else 100
    n = c1.slider("Traces", 1, n_max, min(3, n_max) if is_live else 40, step=1)
    ds_seed = c2.number_input("Dataset seed", min_value=0, value=7, step=1)
    safe = c3.toggle("CP6 safety system", value=False, key="cmp_safe")

    if is_live:
        st.warning(f"On **{brain}** this runs {n}×3 live localizations — keep the count small; it is slow.")

    if st.button("Run comparison", type="primary", disabled=is_live and not ready):
        with st.spinner(f"Running A/B/C on {brain}…"):
            dataset = SyntheticDataset(n=int(n), seed=int(ds_seed), error_ratio=0.0)
            conds = build_conditions(baseline=get_baseline(), live_llm=make_llm(brain, model), safe=safe)
            try:
                reports = compare(dataset, conds)
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")
                return
        rows = [{
            "condition": CONDITION_LABELS[k],
            "top-1": r.top_k.get(1, 0.0),
            "top-3": r.top_k.get(3, 0.0),
            "F1": r.localization_f1,
            "Brier": r.brier,
            "ECE": r.ece,
            "escalation": r.escalation,
        } for k, r in sorted(reports.items())]
        df = pd.DataFrame(rows).set_index("condition")
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
        st.bar_chart(df[["ECE", "Brier"]])
        st.caption(f"Brain: {brain}. Lower ECE / Brier = better calibrated; escalation = abstention rate.")


# --- layout -------------------------------------------------------------------
st.title("🔭 Trace-Reasoner")
st.caption("Multi-agent distributed-trace root-cause localization · CMU Agentic AI capstone")

brain, model, ready = brain_selector()

tab1, tab2 = st.tabs(["Localize a trace", "Compare A/B/C"])
with tab1:
    single_trace_tab(brain, model, ready)
with tab2:
    comparison_tab(brain, model, ready)
