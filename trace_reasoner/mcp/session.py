"""The state and tool logic the MCP server exposes, with no MCP dependency.

This is Checkpoint 4's "memory / state manager" role: a single object that holds the
currently loaded trace and the latency baseline, which the tools read across calls.
Keeping it free of any `mcp` import means it is unit-testable on its own and the rest
of the package does not depend on the MCP SDK being installed.

Observations are returned as JSON strings, matching trace_reasoner.agent.tools_runtime,
so the in-process loop and the MCP front door speak the same shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from trace_reasoner.datasets.synthetic import generate_example, normal_traces
from trace_reasoner.tools.baseline import LatencyBaseline, baseline_latency
from trace_reasoner.tools.walk_tree import survey, walk
from trace_reasoner.trace import Trace


class TraceReasonerSession:
    """Holds the loaded trace and the latency baseline the tools operate on."""

    def __init__(self, baseline: LatencyBaseline | None = None, retriever=None) -> None:
        self._baseline = baseline
        self._retriever = retriever  # a PrecedentRetriever; built lazily for production use
        self._trace: Trace | None = None

    @property
    def baseline(self) -> LatencyBaseline:
        # Built lazily from synthetic normals on first use; pass one in for real data.
        if self._baseline is None:
            self._baseline = LatencyBaseline.from_traces(normal_traces(100, seed=0))
        return self._baseline

    def _require_trace(self) -> Trace:
        if self._trace is None:
            raise ValueError("no trace loaded; call load_synthetic_trace first")
        return self._trace

    # --- tools ---------------------------------------------------------------
    def load_synthetic_trace(self, seed: int = 1, fault: str = "latency") -> str:
        """Load a synthetic trace (with a known injected fault) and return its survey."""
        example = generate_example(seed=seed, fault=fault)
        self._trace = example.trace
        return json.dumps(
            {
                "trace_id": example.trace.trace_id,
                "n_spans": len(example.trace.spans),
                "survey": asdict(survey(example.trace)),
            }
        )

    def survey(self) -> str:
        """Overview of the loaded trace."""
        return json.dumps(asdict(survey(self._require_trace())))

    def walk_tree(self, span_id: str, direction: str = "children") -> str:
        """Navigate the loaded trace from span_id."""
        views = walk(self._require_trace(), span_id, direction)
        return json.dumps([asdict(v) for v in views])

    def baseline_latency(self, span_id: str) -> str:
        """Judge one span's self-time against the historical baseline."""
        verdict = baseline_latency(self.baseline, self._require_trace(), span_id)
        payload = asdict(verdict)
        payload["span_id"] = span_id
        return json.dumps(payload)

    def retrieve_precedents(self, query: str, k: int = 5) -> str:
        """Find past incidents that read like the query (Checkpoint 3 retrieval).

        Builds the production BGE retriever lazily on first use; inject one for tests.
        """
        if self._retriever is None:
            from trace_reasoner.rag.retriever import PrecedentRetriever

            self._retriever = PrecedentRetriever.production()
        from trace_reasoner.rag.retriever import retrieve_precedents as run_retrieve

        return run_retrieve(self._retriever, query, k)
