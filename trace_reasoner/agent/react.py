"""Condition A: the monolithic ReAct localizer.

A constrained ReAct loop (CP2): think -> call one tool -> observe -> repeat, up
to a step budget, then submit ranked root-cause hypotheses. It is a Localizer —
`ReActLocalizer(...)(trace) -> Prediction` — so it plugs into eval.harness.evaluate
exactly like the slowest_leaf baseline. The LLM is injected via LLMClient, so the
same loop runs on Claude or on the deterministic mock.
"""

from __future__ import annotations

from trace_reasoner.agent.llm import (
    AssistantTurn,
    LLMClient,
    ToolResultsMessage,
    UserMessage,
)
from trace_reasoner.agent.tools_runtime import SUBMIT, dispatch, tool_specs
from trace_reasoner.eval.metrics import Hypothesis, Prediction
from trace_reasoner.tools.baseline import LatencyBaseline
from trace_reasoner.tools.walk_tree import survey

SYSTEM_PROMPT = """You localize the root-cause span of an anomaly in a distributed trace, for an on-call SRE.

Work the trace by calling tools: think briefly, call one tool, read the observation, repeat. Start with `survey`. Before naming a span as the cause, confirm it is genuinely anomalous with `baseline_latency` — raw duration is not evidence, since some operations are normally slow. Use `walk_tree` to move through the tree (a slow parent is often waiting on a slow child; an error propagates up from where it originated).

When the evidence points to a cause, call `submit_hypotheses` with up to 3 spans, highest confidence first, confidence in [0,1]. If evidence is weak or the budget runs short, submit your best guess at low confidence rather than nothing — an honest low-confidence answer beats silence."""


class ReActLocalizer:
    def __init__(self, llm: LLMClient, baseline: LatencyBaseline, max_steps: int = 20) -> None:
        self.llm = llm
        self.baseline = baseline
        self.max_steps = max_steps
        self.tools = tool_specs()

    def __call__(self, trace) -> Prediction:
        opening = (
            f"Localize the root-cause span in trace {trace.trace_id}: {len(trace.spans)} spans, "
            f"{round(trace.duration_ms, 1)}ms end-to-end. Start with a survey."
        )
        messages: list = [UserMessage(opening)]

        for _ in range(self.max_steps):
            turn = self.llm.respond(SYSTEM_PROMPT, messages, self.tools)
            messages.append(turn)

            if not turn.tool_calls:
                # Model answered in prose without acting — nudge it back to the tools.
                messages.append(UserMessage("Call a tool, or call submit_hypotheses to finish."))
                continue

            submit = next((c for c in turn.tool_calls if c.name == SUBMIT), None)
            if submit is not None:
                return _to_prediction(trace.trace_id, submit.arguments)

            results = [dispatch(c, trace, self.baseline) for c in turn.tool_calls]
            messages.append(ToolResultsMessage(results))

        # Budget exhausted without a submission — fall back to the hottest self-time spans.
        return _fallback(trace)


def _to_prediction(trace_id: str, arguments: dict) -> Prediction:
    raw = arguments.get("hypotheses", []) if isinstance(arguments, dict) else []
    ranked: list[Hypothesis] = []
    for item in raw:
        if not isinstance(item, dict) or "span_id" not in item:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        rationale = item.get("rationale")
        ranked.append(
            Hypothesis(
                span_id=str(item["span_id"]),
                confidence=confidence,
                evidence=[rationale] if rationale else [],
            )
        )
    ranked.sort(key=lambda h: h.confidence, reverse=True)
    return Prediction(trace_id=trace_id, ranked=ranked)


def _fallback(trace) -> Prediction:
    hot = survey(trace).hottest_by_self_time[:3]
    ranked = [
        Hypothesis(span_id=v.span_id, confidence=0.2, evidence=["fallback: hottest self-time (budget exhausted)"])
        for v in hot
    ]
    return Prediction(trace_id=trace.trace_id, ranked=ranked)
