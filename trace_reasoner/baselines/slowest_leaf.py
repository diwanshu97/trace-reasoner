"""The naive baseline: blame the slowest leaf span.

This is the "Redis is the slowest thing I can see, so Redis is the cause"
heuristic from Checkpoints 2 and 3 — fluent, anchors on the most visible span,
and wrong whenever the real cause is an internal span or an upstream error. It
exists to give the harness a real end-to-end number and a floor for the agent
to beat.
"""

from __future__ import annotations

from trace_reasoner.eval.metrics import Hypothesis, Prediction
from trace_reasoner.trace import Trace


def slowest_leaf(trace: Trace) -> Prediction:
    leaves = sorted(trace.leaves(), key=lambda s: s.duration_ms, reverse=True)
    top = leaves[:5]
    total = sum(s.duration_ms for s in top) or 1.0
    ranked = [
        Hypothesis(
            span_id=s.span_id,
            confidence=s.duration_ms / total,
            evidence=[f"leaf duration {s.duration_ms:.1f}ms"],
        )
        for s in top
    ]
    return Prediction(trace_id=trace.trace_id, ranked=ranked)
