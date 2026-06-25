"""Belief state shared across the multi-agent system (Condition B, Checkpoint 5).

A `Finding` is the structured evidence a specialist writes — span id, fault family,
confidence, and the verified-vs-advisory grounding flag from Checkpoint 3. Specialists
never read each other's findings; they all write into the one list, and only the
synthesizer reads the whole of it. That hub-and-spoke shape is the communication
contract from Checkpoint 5.

`MASState` is the LangGraph graph state. `findings` carries an `operator.add` reducer so
the three specialists' parallel writes merge instead of clobbering each other; every other
key is written by exactly one node per superstep.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from trace_reasoner.eval.metrics import Prediction
from trace_reasoner.trace import Trace

ROLE_NAMES = ("latency", "dependency", "pattern")


@dataclass
class Finding:
    """One specialist's structured verdict on the trace.

    `anomalous` is the latency analyst's grounded baseline result (True/False); the other
    analysts leave it None, since only baseline_latency can decide genuine anomaly. That
    split is what lets the synthesizer prefer verified evidence over advisory evidence.
    """

    role: str
    span_id: str | None
    confidence: float = 0.0  # the specialist's own confidence in [0, 1]
    fault_family: str | None = None
    anomalous: bool | None = None
    evidence: list[str] = field(default_factory=list)


class MASState(TypedDict):
    trace: Trace
    findings: Annotated[list[Finding], operator.add]
    rounds: int
    redispatch: list[str]
    prediction: Prediction | None


def latest_per_role(findings: list[Finding]) -> dict[str, Finding]:
    """Collapse the append-only findings log to the most recent finding per role.

    Re-dispatch appends a fresh finding for a role rather than mutating the old one, so the
    synthesizer always reconciles from each specialist's latest word.
    """
    latest: dict[str, Finding] = {}
    for f in findings:
        latest[f.role] = f
    return latest
