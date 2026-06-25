"""A deterministic, LLM-free client that drives the specialist loops offline.

The analog of agent.mock.HeuristicMockClient, but role-aware: it reads which lens it is from
the system prompt and follows a competent fixed policy for that lens, so the whole Condition B
graph — fan-out, synthesis, and the re-dispatch round — runs in CI with no API key. It is a
fixture, not the real agent.

Policies:
  latency     survey -> baseline_latency on the hottest self-time span -> submit (anomalous flag set)
              on re-dispatch, baseline-check the span named in the synthesizer's hint instead.
  dependency  survey -> walk to the hottest span's subtree -> submit the hottest self-time span.
  pattern     survey -> retrieve_precedents on a hypothesis -> submit (low conf if no precedent).
"""

from __future__ import annotations

import json
import re

from trace_reasoner.agent.llm import AssistantTurn, Message, ToolCall, ToolResultsMessage, ToolSpec, UserMessage


class HeuristicSpecialistClient:
    """Deterministic per-role policy for the three specialists (offline fixture)."""

    def __init__(self) -> None:
        self._n = 0

    def respond(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        self._n += 1
        cid = f"mas-{self._n}"
        role = _role_of(system)
        last = messages[-1]

        # Opening move: everyone surveys first.
        if isinstance(last, UserMessage):
            hint_span = _hint_span(last.text)
            if role == "latency" and hint_span:
                # Re-dispatch: baseline-check the span the synthesizer pointed at.
                return AssistantTurn("Baseline-checking the flagged span.",
                                     [ToolCall(cid, "baseline_latency", {"span_id": hint_span})])
            return AssistantTurn("Surveying.", [ToolCall(cid, "survey", {})])

        if isinstance(last, ToolResultsMessage) and last.results:
            data = _safe_json(last.results[-1].content)

            # Just surveyed → take the lens-specific next action.
            if isinstance(data, dict) and data.get("hottest_by_self_time"):
                hottest = data["hottest_by_self_time"][0]["span_id"]
                family = "saturation"
                if role == "latency":
                    return AssistantTurn(f"Checking {hottest}.",
                                         [ToolCall(cid, "baseline_latency", {"span_id": hottest})])
                if role == "dependency":
                    return AssistantTurn(f"Walking children of {hottest}.",
                                         [ToolCall(cid, "walk_tree", {"span_id": hottest, "direction": "children"})])
                if role == "pattern":
                    return AssistantTurn("Searching precedents.",
                                         [ToolCall(cid, "retrieve_precedents",
                                                   {"query": "high self-time span causing latency spike", "k": 5})])

            # Latency: baseline verdict came back → submit it with the anomalous flag.
            if isinstance(data, dict) and "is_anomalous" in data:
                span_id = data.get("span_id")
                anom = bool(data.get("is_anomalous"))
                return AssistantTurn("Submitting latency finding.",
                                     [ToolCall(cid, "submit_finding", {
                                         "span_id": span_id,
                                         "confidence": 0.85 if anom else 0.4,
                                         "fault_family": "saturation",
                                         "anomalous": anom,
                                         "rationale": "hottest self-time, baseline-checked",
                                     })])

            # Dependency: walked children → submit the hottest self-time span we can see.
            if isinstance(data, list):
                if data:
                    target = max(data, key=lambda v: v.get("self_time_ms", 0.0))
                    span_id = target["span_id"]
                else:
                    span_id = None
                return AssistantTurn("Submitting dependency finding.",
                                     [ToolCall(cid, "submit_finding", {
                                         "span_id": span_id,
                                         "confidence": 0.6 if span_id else 0.1,
                                         "fault_family": "dependency",
                                         "rationale": "structural origin by self-time in subtree",
                                     })])

            # Pattern: precedents came back → submit with confidence keyed on whether any cleared the floor.
            if isinstance(data, dict) and ("precedents" in data or data.get("result")):
                precedents = data.get("precedents") or []
                if precedents:
                    fam = precedents[0].get("fault_family") or "saturation"
                    return AssistantTurn("Submitting pattern finding.",
                                         [ToolCall(cid, "submit_finding", {
                                             "confidence": 0.55,
                                             "fault_family": fam,
                                             "rationale": f"matches precedent {precedents[0].get('id', '?')}",
                                         })])
                return AssistantTurn("No precedent.",
                                     [ToolCall(cid, "submit_finding", {
                                         "confidence": 0.1,
                                         "rationale": "no precedent retrieved",
                                     })])

        # Fallback: submit an empty finding to end cleanly.
        return AssistantTurn("No lead.", [ToolCall(cid, "submit_finding", {"confidence": 0.0})])


def _role_of(system: str) -> str:
    head = system[:40].upper()
    if "LATENCY" in head:
        return "latency"
    if "DEPENDENCY" in head:
        return "dependency"
    if "PATTERN" in head:
        return "pattern"
    return "latency"


def _hint_span(text: str) -> str | None:
    m = re.search(r"implicates span (\S+?)[\s)]", text)
    return m.group(1) if m else None


def _safe_json(text: str):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None
