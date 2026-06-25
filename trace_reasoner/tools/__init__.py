"""The agent's tool surface over the Trace model.

Written once as plain Python so both the research agent loop and the future MCP
wrappers (for the demo) call the same code. These are the CP2 tools:

  walk_tree       — navigate the span tree (children/parent/siblings/ancestors)
  survey          — the opening trace overview (hottest spans, critical path)
  baseline_latency— "is this span normally this slow?" against a real baseline
"""

from trace_reasoner.tools.walk_tree import SpanView, TraceSurvey, survey, view, walk
from trace_reasoner.tools.baseline import (
    LatencyBaseline,
    LatencyStats,
    Verdict,
    baseline_latency,
)

__all__ = [
    "SpanView",
    "TraceSurvey",
    "survey",
    "view",
    "walk",
    "LatencyBaseline",
    "LatencyStats",
    "Verdict",
    "baseline_latency",
]
