"""Guardrails: the static + verification layer of the Checkpoint 6 control system.

These are the constraints that bound the agent *before* it answers (input checks) and *after*
it answers (source verification, output schema), independent of how confident it is. They are the
"catch what evaluation cannot predict offline" layer from the CP6 plan — injection, hallucination,
oversized input — so they are deterministic, cheap, and never call the model.

  check_input        size cap, prompt-injection scan, and secret redaction on the trace, which
                     carries bug- or attacker-controlled text (span names, logs) into context.
  verify_prediction  every named span must resolve to a real span in THIS trace; a hallucinated
                     span is dropped, not ranked. Also enforces the output schema (sorted,
                     clamped, top-3) so a localizer can never emit a malformed or bare answer.
  groundedness       the share of an agent's named spans that resolve to real spans — the inverse
                     hallucination rate, a CP6 metric.
  ToolPolicy         least privilege: read-only tools are allowed; a side-effecting tool
                     (rerun_scenario) needs explicit approval and a sandbox, never production.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from trace_reasoner.eval.metrics import Hypothesis, Prediction
from trace_reasoner.trace import Trace

DEFAULT_MAX_SPANS = 2000

# Instruction-like text appearing in span attributes/operations is data masquerading as a command
# (MAESTRO "attacks on mind and memory"). We flag, never execute — the trace is data, not a prompt.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |the )?(previous|prior|above)", re.I),
    re.compile(r"disregard (all |the )?(previous|prior|instructions)", re.I),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"\bsystem prompt\b", re.I),
    re.compile(r"<\s*/?\s*(system|assistant|tool)\b", re.I),
]
# Common secret shapes; redacted before the text ever reaches the model.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*[=:]\s*\S+"),
]


@dataclass
class InputReport:
    """Verdict of the pre-flight input guardrails."""

    ok: bool
    n_spans: int
    violations: list[str] = field(default_factory=list)
    redactions: list[str] = field(default_factory=list)  # span_ids whose attributes were redacted


def _scan_text(*values: str) -> bool:
    return any(p.search(v) for v in values if v for p in _INJECTION_PATTERNS)


def check_input(trace: Trace, max_spans: int = DEFAULT_MAX_SPANS) -> InputReport:
    """Pre-flight checks on the untrusted trace: size cap, injection scan, secret redaction.

    A size violation or a detected injection makes the report not-ok — the SafeLocalizer halts
    and escalates rather than feeding the trace to the agent. Secret redaction is recorded but is
    not on its own a halt (the sanitized trace is still analyzable).
    """
    violations: list[str] = []
    redactions: list[str] = []

    n = len(trace.spans)
    if n > max_spans:
        violations.append(f"trace has {n} spans, over the {max_spans} cap")

    for s in trace.spans:
        attr_values = list(s.attributes.values())
        if _scan_text(s.operation, s.service, *attr_values):
            violations.append(f"possible prompt injection in span {s.span_id}")
        if any(p.search(v) for v in attr_values for p in _SECRET_PATTERNS):
            redactions.append(s.span_id)

    return InputReport(ok=not violations, n_spans=n, violations=violations, redactions=redactions)


def redact_secrets(trace: Trace) -> int:
    """Mask secret-shaped attribute values in place; return how many were redacted.

    Mutates span attributes so the secret never reaches the model. Returns the count so the caller
    can log it (runtime monitoring). Span structure and timings are untouched.
    """
    redacted = 0
    for s in trace.spans:
        for key, value in list(s.attributes.items()):
            if any(p.search(value) for p in _SECRET_PATTERNS):
                s.attributes[key] = "[REDACTED]"
                redacted += 1
    return redacted


def verify_prediction(prediction: Prediction, trace: Trace) -> tuple[Prediction, list[str]]:
    """Source-verify and schema-enforce a localizer's output.

    Drops any hypothesis whose span_id does not resolve to a real span in this trace (a
    hallucination is rejected, not ranked — the CP6 source-verification guardrail), clamps each
    confidence to [0, 1], sorts descending, and caps at top-3. Returns the cleaned Prediction and
    the list of dropped (hallucinated) span_ids.
    """
    real = {s.span_id for s in trace.spans}
    kept: list[Hypothesis] = []
    dropped: list[str] = []
    for h in prediction.ranked:
        if h.span_id not in real:
            dropped.append(h.span_id)
            continue
        kept.append(
            Hypothesis(
                span_id=h.span_id,
                confidence=min(1.0, max(0.0, h.confidence)),
                fault_family=h.fault_family,
                evidence=h.evidence,
            )
        )
    kept.sort(key=lambda h: h.confidence, reverse=True)
    return Prediction(trace_id=prediction.trace_id, ranked=kept[:3]), dropped


def groundedness(prediction: Prediction, trace: Trace) -> float:
    """Fraction of named spans that resolve to a real span in the trace (inverse hallucination).

    An abstention (empty prediction) fabricates nothing, so it is perfectly grounded (1.0).
    """
    if not prediction.ranked:
        return 1.0
    real = {s.span_id for s in trace.spans}
    resolved = sum(1 for h in prediction.ranked if h.span_id in real)
    return resolved / len(prediction.ranked)


@dataclass
class Authorization:
    allowed: bool
    needs_approval: bool
    reason: str


class ToolPolicy:
    """Least-privilege tool access (CP6 guardrail).

    Read-only tools (the ones the specialists and the beam critic use) are allowed outright. A
    side-effecting tool — rerun_scenario, which replays a scenario — is gated: it needs explicit
    approval and runs only against a sandbox, never production. The policy is enforced at the tool
    boundary so an agent cannot escalate its own privileges.
    """

    READ_ONLY = frozenset({"survey", "walk_tree", "baseline_latency", "retrieve_precedents"})
    SIDE_EFFECTING = frozenset({"rerun_scenario"})

    def authorize(self, tool: str, approved: bool = False) -> Authorization:
        if tool in self.READ_ONLY:
            return Authorization(True, False, "read-only")
        if tool in self.SIDE_EFFECTING:
            if approved:
                return Authorization(True, True, "side-effecting: approved, sandbox only")
            return Authorization(False, True, "side-effecting: requires human approval")
        return Authorization(False, False, f"unknown tool {tool!r}: denied by default")
