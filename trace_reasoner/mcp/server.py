"""FastMCP server exposing Trace-Reasoner's tools over MCP.

Run it (stdio transport):
    pip install -e '.[mcp]'
    python -m trace_reasoner.mcp

Register it with an MCP client (e.g. Claude Desktop / Claude Code) by pointing the
client at `python -m trace_reasoner.mcp`. The four tools below are thin wrappers over
TraceReasonerSession, which holds the loaded trace and baseline (the shared state).
A typical client flow: load_synthetic_trace -> survey -> walk_tree / baseline_latency.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from trace_reasoner.mcp.session import TraceReasonerSession

mcp = FastMCP("trace-reasoner")
_session = TraceReasonerSession()


@mcp.tool()
def load_synthetic_trace(seed: int = 1, fault: str = "latency") -> str:
    """Load a synthetic trace with a known injected fault into the session and return its survey.

    fault is one of "latency", "error", or "none". Call this first.
    """
    return _session.load_synthetic_trace(seed=seed, fault=fault)


@mcp.tool()
def survey() -> str:
    """Overview of the loaded trace: end-to-end latency, services, error-span count,
    the hottest spans by self-time, and the critical path."""
    return _session.survey()


@mcp.tool()
def walk_tree(span_id: str, direction: str = "children") -> str:
    """Navigate the loaded trace from span_id.

    direction is one of: node, children, parent, siblings, ancestors.
    """
    return _session.walk_tree(span_id, direction)


@mcp.tool()
def baseline_latency(span_id: str) -> str:
    """Check whether a span's self-time is anomalous versus the historical
    per-(service, operation) baseline. Use this to confirm a span is genuinely slow."""
    return _session.baseline_latency(span_id)


@mcp.tool()
def retrieve_precedents(query: str, k: int = 5) -> str:
    """Find past incidents (public postmortems, chaos-experiment scenarios) that read like
    the query, as analogical evidence. Returns up to k precedents above a 0.55 similarity
    floor, or "no precedent retrieved" when nothing clears it."""
    return _session.retrieve_precedents(query, k)


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
