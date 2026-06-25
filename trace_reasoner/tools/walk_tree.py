"""walk_tree + survey: how the agent navigates and reads a trace.

These return small structured views (not raw Span objects) so the same output
serialises cleanly whether it feeds an in-process ReAct loop or an MCP tool call.
"""

from __future__ import annotations

from dataclasses import dataclass

from trace_reasoner.trace import Trace

_DIRECTIONS = ("node", "children", "parent", "siblings", "ancestors")


@dataclass
class SpanView:
    span_id: str
    service: str
    operation: str
    duration_ms: float
    self_time_ms: float
    status: str
    n_children: int


def view(trace: Trace, span_id: str) -> SpanView:
    s = trace.get(span_id)
    return SpanView(
        span_id=s.span_id,
        service=s.service,
        operation=s.operation,
        duration_ms=round(s.duration_ms, 3),
        self_time_ms=round(trace.self_time_ms(span_id), 3),
        status=s.status,
        n_children=len(trace.children(span_id)),
    )


def walk(trace: Trace, span_id: str, direction: str = "children") -> list[SpanView]:
    """Move around the span tree from `span_id`.

    direction: "node" (just this span), "children", "parent", "siblings",
    "ancestors" (parent -> root order).
    """
    if direction == "node":
        return [view(trace, span_id)]
    if direction == "children":
        return [view(trace, c.span_id) for c in trace.children(span_id)]
    if direction == "parent":
        p = trace.parent(span_id)
        return [view(trace, p.span_id)] if p else []
    if direction == "ancestors":
        return [view(trace, a.span_id) for a in trace.ancestors(span_id)]
    if direction == "siblings":
        p = trace.parent(span_id)
        if p is None:
            return []
        return [view(trace, c.span_id) for c in trace.children(p.span_id) if c.span_id != span_id]
    raise ValueError(f"unknown direction {direction!r}; choose from {_DIRECTIONS}")


@dataclass
class TraceSurvey:
    trace_id: str
    n_spans: int
    duration_ms: float
    services: list[str]
    error_spans: int
    hottest_by_self_time: list[SpanView]
    critical_path: list[SpanView]


def survey(trace: Trace, top: int = 5) -> TraceSurvey:
    """The opening overview the agent reads before forming a hypothesis.

    Surfaces the spans that consume the most exclusive time (the natural first
    suspects) and the latency-gating critical path.
    """
    by_self = sorted(trace.spans, key=lambda s: trace.self_time_ms(s.span_id), reverse=True)
    return TraceSurvey(
        trace_id=trace.trace_id,
        n_spans=len(trace.spans),
        duration_ms=round(trace.duration_ms, 3),
        services=sorted({s.service for s in trace.spans}),
        error_spans=sum(1 for s in trace.spans if s.is_error),
        hottest_by_self_time=[view(trace, s.span_id) for s in by_self[:top]],
        critical_path=[view(trace, s.span_id) for s in trace.critical_path()],
    )
