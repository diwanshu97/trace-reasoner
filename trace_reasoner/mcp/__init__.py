"""MCP server: the demo front door over the same Python tools the research loop uses.

The research core (trace_reasoner.agent) calls trace_reasoner.tools directly, in an
instrumented Python loop, so the iso-token ablation and DPO trajectory capture stay
under our control. This package exposes those exact tools over MCP for an external
client (the demo GUI, or any MCP client) to drive. Same logic, a second front door.

The tool logic lives in TraceReasonerSession (no `mcp` import, fully unit-tested).
The FastMCP server in server.py is a thin registration layer over it, so importing
this package never requires the `mcp` SDK; only running the server does.
"""

from trace_reasoner.mcp.session import TraceReasonerSession

__all__ = ["TraceReasonerSession"]
