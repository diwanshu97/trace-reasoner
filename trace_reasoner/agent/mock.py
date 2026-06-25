"""A deterministic, LLM-free client that drives the ReAct loop.

Policy: survey -> baseline_latency on the hottest self-time span -> submit it.
That happens to be a competent localizer for latency faults (the injected span
has inflated self-time), so it both exercises the full loop offline and serves
as a no-API agent for CI. It is NOT the real agent — it's a fixture.
"""

from __future__ import annotations

import json

from trace_reasoner.agent.llm import (
    AssistantTurn,
    Message,
    ToolCall,
    ToolResultsMessage,
    ToolSpec,
    UserMessage,
)


class HeuristicMockClient:
    def __init__(self) -> None:
        self._n = 0

    def respond(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        self._n += 1
        call_id = f"mock-{self._n}"
        last = messages[-1]

        # Opening move: survey the trace.
        if isinstance(last, UserMessage) and len(messages) == 1:
            return AssistantTurn(
                text="Surveying the trace.",
                tool_calls=[ToolCall(call_id, "survey", {})],
            )

        if isinstance(last, ToolResultsMessage) and last.results:
            data = _safe_json(last.results[-1].content)

            # After a survey: check the hottest self-time span against baseline.
            if isinstance(data, dict) and data.get("hottest_by_self_time"):
                span_id = data["hottest_by_self_time"][0]["span_id"]
                return AssistantTurn(
                    text=f"Checking {span_id} against its baseline.",
                    tool_calls=[ToolCall(call_id, "baseline_latency", {"span_id": span_id})],
                )

            # After a baseline verdict: submit that span as the hypothesis.
            if isinstance(data, dict) and "is_anomalous" in data:
                span_id = data.get("span_id")
                confidence = 0.85 if data.get("is_anomalous") else 0.4
                return AssistantTurn(
                    text="Submitting the root-cause hypothesis.",
                    tool_calls=[
                        ToolCall(
                            call_id,
                            "submit_hypotheses",
                            {
                                "hypotheses": [
                                    {
                                        "span_id": span_id,
                                        "confidence": confidence,
                                        "rationale": "hottest self-time, baseline-checked",
                                    }
                                ]
                            },
                        )
                    ],
                )

        # Shouldn't normally reach here — submit empty to end cleanly.
        return AssistantTurn(
            text="No further leads.",
            tool_calls=[ToolCall(call_id, "submit_hypotheses", {"hypotheses": []})],
        )


def _safe_json(text: str):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None
