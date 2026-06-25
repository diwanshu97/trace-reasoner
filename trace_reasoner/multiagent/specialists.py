"""The three specialist analysts of Condition B (Checkpoint 5).

Each analyst is the same constrained ReAct loop as Condition A, but prompt-scoped to one
lens and handed only the tools that lens needs:

  latency     survey + baseline_latency   — is a span genuinely anomalous vs its baseline?
  dependency  survey + walk_tree          — is a slow span the cause, or blocked on a child?
  pattern     survey + retrieve_precedents — have we seen this failure shape before?

Giving each its own context (and only its own tools) is the whole point of the decomposition:
a loud raw-latency signal in one lens cannot crowd out a subtle structural one in another, the
single-context failure mode from Checkpoint 2. Every analyst ends by calling `submit_finding`,
which the loop turns into a structured `Finding` in shared state. The LLM is injected via the
same `LLMClient` protocol Condition A uses, so a specialist runs on Claude or on the offline mock
unchanged — and its token usage is counted on the same ruler, keeping the iso-budget ablation honest.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from trace_reasoner.agent.llm import (
    AssistantTurn,
    LLMClient,
    Message,
    ToolResult,
    ToolResultsMessage,
    ToolSpec,
    UserMessage,
)
from trace_reasoner.multiagent.state import Finding
from trace_reasoner.rag.retriever import PrecedentRetriever
from trace_reasoner.rag.retriever import retrieve_precedents as run_retrieve
from trace_reasoner.tools.baseline import LatencyBaseline, baseline_latency
from trace_reasoner.tools.walk_tree import survey, walk
from trace_reasoner.trace import Trace

SUBMIT_FINDING = "submit_finding"

_SURVEY_SPEC = ToolSpec(
    name="survey",
    description="Overview of the trace: end-to-end latency, services, error-span count, hottest spans by self-time, and the critical path. Call this first.",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
)
_BASELINE_SPEC = ToolSpec(
    name="baseline_latency",
    description="Check whether a span's self-time is anomalous versus the historical per-(service, operation) baseline. Raw duration alone is not evidence.",
    input_schema={
        "type": "object",
        "properties": {"span_id": {"type": "string"}},
        "required": ["span_id"],
        "additionalProperties": False,
    },
)
_WALK_SPEC = ToolSpec(
    name="walk_tree",
    description="Navigate the span tree from a span_id. direction is one of node | children | parent | siblings | ancestors.",
    input_schema={
        "type": "object",
        "properties": {
            "span_id": {"type": "string"},
            "direction": {"type": "string", "enum": ["node", "children", "parent", "siblings", "ancestors"]},
        },
        "required": ["span_id", "direction"],
        "additionalProperties": False,
    },
)
_RETRIEVE_SPEC = ToolSpec(
    name="retrieve_precedents",
    description="Find past incidents that read like a short hypothesis. Returns causal analogs with their resolved fault family, or 'no precedent retrieved'.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)


def _submit_spec(role: str) -> ToolSpec:
    props = {
        "span_id": {"type": "string", "description": "the span you implicate, or omit if none"},
        "confidence": {"type": "number", "description": "your confidence in [0,1]"},
        "fault_family": {"type": "string", "enum": list(_FAULT_FAMILIES)},
        "rationale": {"type": "string"},
    }
    if role == "latency":
        props["anomalous"] = {
            "type": "boolean",
            "description": "did baseline_latency confirm the span is genuinely anomalous?",
        }
    return ToolSpec(
        name=SUBMIT_FINDING,
        description="Submit your single best finding for your lens and end your analysis.",
        input_schema={"type": "object", "properties": props, "required": ["confidence"], "additionalProperties": False},
    )


_FAULT_FAMILIES = ("timeout", "contention", "saturation", "dependency", "configuration", "exception")

_PROMPTS = {
    "latency": (
        "You are the LATENCY analyst on an incident-response team localizing the root-cause span of an "
        "anomaly in one distributed trace. Your lens: is a span GENUINELY anomalous against its historical "
        "baseline? Start with `survey`, then call `baseline_latency` on the suspects — raw duration is not "
        "evidence, some operations are normally slow. Report the span that is anomalous and explains the most "
        "excess wall-time. Call `submit_finding` with anomalous=true only if baseline_latency confirmed it."
    ),
    "dependency": (
        "You are the DEPENDENCY analyst on an incident-response team localizing the root-cause span of an "
        "anomaly in one distributed trace. Your lens: is a slow span the actual cause, or is it merely blocked "
        "on a slow child, or an error propagating up from below? Start with `survey`, then use `walk_tree` "
        "(children/parent/ancestors) to find where the cost or the error truly originates. Submit the span you "
        "believe is the structural origin, not its victims."
    ),
    "pattern": (
        "You are the PATTERN analyst on an incident-response team localizing the root-cause span of an anomaly "
        "in one distributed trace. Your lens: have we seen this failure SHAPE before? Start with `survey`, form "
        "a one-line hypothesis, and call `retrieve_precedents` with it. Past incidents that look like this often "
        "reveal the real cause is a hop away from the obvious slow span (e.g. connection-pool saturation, not the "
        "Redis call itself). If nothing clears the floor, say so and submit low confidence rather than guessing."
    ),
}

_TOOLS = {
    "latency": [_SURVEY_SPEC, _BASELINE_SPEC],
    "dependency": [_SURVEY_SPEC, _WALK_SPEC],
    "pattern": [_SURVEY_SPEC, _RETRIEVE_SPEC],
}


def _dispatch(call, trace: Trace, baseline: LatencyBaseline, retriever: PrecedentRetriever | None) -> ToolResult:
    """Execute one specialist tool call, returning its observation as JSON (matches tools_runtime)."""
    try:
        if call.name == "survey":
            return ToolResult(call.id, json.dumps(asdict(survey(trace))))
        if call.name == "walk_tree":
            views = walk(trace, call.arguments["span_id"], call.arguments.get("direction", "children"))
            return ToolResult(call.id, json.dumps([asdict(v) for v in views]))
        if call.name == "baseline_latency":
            span_id = call.arguments["span_id"]
            payload = asdict(baseline_latency(baseline, trace, span_id))
            payload["span_id"] = span_id
            return ToolResult(call.id, json.dumps(payload))
        if call.name == "retrieve_precedents":
            if retriever is None:
                return ToolResult(call.id, json.dumps({"result": "retriever unavailable", "precedents": []}))
            return ToolResult(call.id, run_retrieve(retriever, call.arguments["query"], int(call.arguments.get("k", 5))))
        return ToolResult(call.id, f"unknown tool {call.name!r}", is_error=True)
    except KeyError as exc:
        return ToolResult(call.id, f"bad arguments or unknown span_id: {exc}", is_error=True)
    except ValueError as exc:
        return ToolResult(call.id, f"invalid argument: {exc}", is_error=True)


def run_specialist(
    role: str,
    llm: LLMClient,
    trace: Trace,
    baseline: LatencyBaseline,
    retriever: PrecedentRetriever | None = None,
    max_steps: int = 8,
    hint: str | None = None,
) -> Finding:
    """Run one specialist's bounded ReAct loop and return its structured Finding.

    `hint` is the synthesizer's targeted re-dispatch question (Checkpoint 5) — e.g. asking
    the latency analyst to baseline-check the span the dependency analyst implicated. It is
    appended to the opening so the specialist re-runs its lens aimed at the reframed question.
    """
    if role not in _PROMPTS:
        raise ValueError(f"unknown specialist role {role!r}; choose from {tuple(_PROMPTS)}")

    tools = [*_TOOLS[role], _submit_spec(role)]
    system = _PROMPTS[role]
    opening = (
        f"Localize the root-cause span in trace {trace.trace_id}: {len(trace.spans)} spans, "
        f"{round(trace.duration_ms, 1)}ms end-to-end. Work your lens, then submit one finding."
    )
    if hint:
        opening += f"\n\nFocus from the synthesizer: {hint}"
    messages: list[Message] = [UserMessage(opening)]

    for _ in range(max_steps):
        turn = llm.respond(system, messages, tools)
        messages.append(turn)

        if not turn.tool_calls:
            messages.append(UserMessage(f"Call a tool, or call {SUBMIT_FINDING} to finish."))
            continue

        submit = next((c for c in turn.tool_calls if c.name == SUBMIT_FINDING), None)
        if submit is not None:
            return _to_finding(role, submit.arguments)

        results = [_dispatch(c, trace, baseline, retriever) for c in turn.tool_calls]
        messages.append(ToolResultsMessage(results))

    # Budget exhausted without a submission — an honest empty finding for its lens.
    return Finding(role=role, span_id=None, confidence=0.0, evidence=["budget exhausted, no finding"])


def _to_finding(role: str, args: dict) -> Finding:
    args = args if isinstance(args, dict) else {}
    try:
        confidence = min(1.0, max(0.0, float(args.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    span_id = args.get("span_id") or None
    rationale = args.get("rationale")
    return Finding(
        role=role,
        span_id=str(span_id) if span_id else None,
        confidence=confidence,
        fault_family=args.get("fault_family"),
        anomalous=args.get("anomalous") if role == "latency" else None,
        evidence=[rationale] if rationale else [],
    )
