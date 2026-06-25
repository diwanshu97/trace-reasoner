"""Condition B wired as a LangGraph state machine (Module 5 / Checkpoint 5).

The coordination strategy from Checkpoint 5, made executable:

    START -> {latency, dependency, pattern}   (fan-out, parallel, isolated contexts)
                \\        |         /
                 \\       |        /
                  -->  synthesizer  -->  END
                          ^  |
                          |  v  (capped, targeted)
                    re-dispatch one specialist

The three specialists fan out from START and write structured Findings into the shared belief
state (`findings`, merged by an operator.add reducer). Each surveys the trace in its own isolated
context (its first tool call), exactly as the Condition A loop does — there is no shared survey
node, so a loud signal in one lens cannot leak into another. The synthesizer is the only node that
reads all findings; it reconciles a ranked Prediction and decides whether one targeted re-dispatch
is worth a round. Specialists never edge to each other — that is the hub-and-spoke communication
contract. LangGraph owns the execution flow; the synthesizer stays plain Python so the iso-token
budget is unaffected by orchestration.

`MultiAgentLocalizer` is a Localizer — `mas(trace) -> Prediction` — so it plugs into
eval.harness.evaluate exactly like the Condition A ReAct loop and the slowest_leaf baseline.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from trace_reasoner.agent.llm import LLMClient
from trace_reasoner.eval.metrics import Prediction
from trace_reasoner.multiagent.specialists import run_specialist
from trace_reasoner.multiagent.state import MASState
from trace_reasoner.multiagent.synthesizer import reconcile, redispatch_hint, redispatch_targets
from trace_reasoner.tools.baseline import LatencyBaseline
from trace_reasoner.trace import Trace


def build_graph(
    llm: LLMClient,
    baseline: LatencyBaseline,
    retriever=None,
    max_rounds: int = 2,
    specialist_max_steps: int = 8,
):
    """Compile the Condition B graph. Dependencies are bound here; the trace flows in via state."""

    def specialist_node(role: str):
        def node(state: MASState) -> dict:
            hint = None
            if role in state.get("redispatch", []):
                hint = redispatch_hint(state["findings"])
            finding = run_specialist(
                role,
                llm,
                state["trace"],
                baseline,
                retriever=retriever,
                max_steps=specialist_max_steps,
                hint=hint,
            )
            return {"findings": [finding]}

        return node

    def synth_node(state: MASState) -> dict:
        rounds = state.get("rounds", 0) + 1
        prediction = reconcile(state["trace"].trace_id, state["findings"])
        targets = redispatch_targets(state["findings"]) if rounds < max_rounds else []
        return {"rounds": rounds, "prediction": prediction, "redispatch": targets}

    def route(state: MASState):
        # Re-dispatch the targeted specialists, or finish.
        return state.get("redispatch") or END

    g = StateGraph(MASState)
    g.add_node("latency", specialist_node("latency"))
    g.add_node("dependency", specialist_node("dependency"))
    g.add_node("pattern", specialist_node("pattern"))
    g.add_node("synthesizer", synth_node)

    for role in ("latency", "dependency", "pattern"):
        g.add_edge(START, role)  # fan-out: all three specialists start in parallel
        g.add_edge(role, "synthesizer")
    g.add_conditional_edges(
        "synthesizer",
        route,
        {"latency": "latency", "dependency": "dependency", "pattern": "pattern", END: END},
    )
    return g.compile()


class MultiAgentLocalizer:
    """Condition B: specialist analysts + synthesizer, orchestrated with LangGraph.

    A Localizer like ReActLocalizer — it scores on the same harness, so the A/B/C ablation is
    measured on one ruler. The LLM is injected (Claude or the offline mock), as is the latency
    baseline and the optional precedent retriever (the pattern analyst degrades gracefully without it).
    """

    def __init__(
        self,
        llm: LLMClient,
        baseline: LatencyBaseline,
        retriever=None,
        max_rounds: int = 2,
        specialist_max_steps: int = 8,
    ) -> None:
        self._app = build_graph(llm, baseline, retriever, max_rounds, specialist_max_steps)

    def __call__(self, trace: Trace) -> Prediction:
        final = self._app.invoke(
            {
                "trace": trace,
                "findings": [],
                "rounds": 0,
                "redispatch": [],
                "prediction": None,
            }
        )
        prediction = final.get("prediction")
        return prediction if prediction is not None else Prediction(trace_id=trace.trace_id, ranked=[])
