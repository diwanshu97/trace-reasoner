"""The tool surface the agent sees, and the dispatcher that executes calls.

Wraps the pure tools in trace_reasoner.tools as LLM-callable tools. Observations
are returned as JSON strings so they serialise identically for the model and for
the deterministic mock. `submit_hypotheses` is the terminal tool — the loop
catches it and turns its arguments into the final Prediction.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from trace_reasoner.agent.llm import ToolCall, ToolResult, ToolSpec
from trace_reasoner.tools.baseline import LatencyBaseline, baseline_latency
from trace_reasoner.tools.walk_tree import survey, walk
from trace_reasoner.trace import Trace

SUBMIT = "submit_hypotheses"


def tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="survey",
            description=(
                "Overview of the trace: end-to-end latency, services, error-span "
                "count, the hottest spans by self-time, and the critical path. "
                "Call this first."
            ),
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            name="walk_tree",
            description=(
                "Navigate the span tree from a span_id. direction is one of "
                "node | children | parent | siblings | ancestors."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "span_id": {"type": "string"},
                    "direction": {
                        "type": "string",
                        "enum": ["node", "children", "parent", "siblings", "ancestors"],
                    },
                },
                "required": ["span_id", "direction"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="baseline_latency",
            description=(
                "Check whether a span's self-time is anomalous versus the historical "
                "per-(service, operation) baseline. Use this to confirm a span is "
                "genuinely slow — raw duration alone is not evidence."
            ),
            input_schema={
                "type": "object",
                "properties": {"span_id": {"type": "string"}},
                "required": ["span_id"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name=SUBMIT,
            description=(
                "Submit the final ranked root-cause hypotheses and end the "
                "investigation. Highest-confidence first; confidence in [0,1]."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "span_id": {"type": "string"},
                                "confidence": {"type": "number"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["span_id", "confidence"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["hypotheses"],
                "additionalProperties": False,
            },
        ),
    ]


def dispatch(call: ToolCall, trace: Trace, baseline: LatencyBaseline) -> ToolResult:
    """Execute a (non-submit) tool call and return its observation as JSON."""
    try:
        if call.name == "survey":
            return ToolResult(call.id, json.dumps(asdict(survey(trace))))

        if call.name == "walk_tree":
            views = walk(trace, call.arguments["span_id"], call.arguments.get("direction", "children"))
            return ToolResult(call.id, json.dumps([asdict(v) for v in views]))

        if call.name == "baseline_latency":
            span_id = call.arguments["span_id"]
            verdict = baseline_latency(baseline, trace, span_id)
            payload = asdict(verdict)
            payload["span_id"] = span_id  # echo so the caller knows which span this verdict is for
            return ToolResult(call.id, json.dumps(payload))

        return ToolResult(call.id, f"unknown tool {call.name!r}", is_error=True)
    except KeyError as exc:
        return ToolResult(call.id, f"bad arguments or unknown span_id: {exc}", is_error=True)
    except ValueError as exc:
        return ToolResult(call.id, f"invalid argument: {exc}", is_error=True)
