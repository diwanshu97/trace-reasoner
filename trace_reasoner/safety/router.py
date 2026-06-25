"""The trust/risk router and SafeLocalizer: the Checkpoint 6 control system, composed.

CP6's three layers, wired together around any condition (A, B, or C):

    trace -> [input guardrails] -> inner localizer -> [source verification] -> [trust/risk router]

The router is the deck's "router, not gate" model: instead of asking "has a human approved this?",
it asks "does this output meet the criteria to proceed?" and assigns a lane:

    AUTO      confident, in-distribution, grounded        -> deliver as-is
    REVIEW    answer stands but is flagged for a human     -> deliver, annotated
    ESCALATE  too risky to answer autonomously             -> abstain, route to the on-call SRE

The risk signals are the ones CP6 names: top-1 confidence vs a calibrated floor, the margin to the
runner-up, an out-of-distribution signal grounded in the latency baseline, and a guardrail trip
(injection / oversized input / hallucinated spans). ESCALATE empties the Prediction, so it counts
in `escalation_rate` and the abstention shows up on the same accuracy-coverage curve as the
"inconclusive within budget" path the agents already produce — one ruler for the whole system.

`SafeLocalizer` wraps an inner Localizer and is itself a Localizer, so it scores on the existing
harness unchanged; `.decide(trace)` exposes the full RoutedDecision for inspection and the
safety-specific metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from trace_reasoner.eval.metrics import Prediction
from trace_reasoner.safety.guardrails import (
    check_input,
    groundedness,
    redact_secrets,
    verify_prediction,
)
from trace_reasoner.tools.baseline import LatencyBaseline
from trace_reasoner.tools.walk_tree import survey
from trace_reasoner.trace import Trace


class Lane(str, Enum):
    AUTO = "auto"
    REVIEW = "review"
    ESCALATE = "escalate"


@dataclass
class RouterThresholds:
    """The operating point. Tuned against the accuracy-coverage curve (CP6): too conservative
    over-escalates until the SRE ignores the tool; too permissive lets confident-wrong through.

    These defaults were chosen empirically on the offline build: the conditions self-report
    confidence on different scales (ReAct ~0.85, multi-agent ~0.93, Tree-of-Thought 0.42-0.75),
    and the accuracy-coverage curve stays at 100% top-1 down to ~0.5 coverage, so the floor sits
    just under that knee. The earlier 0.70 ceiling sat above almost the entire ToT range and
    swept every C answer into REVIEW; 0.60 restores a real AUTO/REVIEW/ESCALATE spread while still
    catching the genuinely low-confidence and out-of-distribution cases. Re-tune for live Claude,
    whose confidence distribution differs from the mock's."""

    confidence_floor: float = 0.48   # below this top-1 confidence -> escalate
    review_ceiling: float = 0.60     # between floor and this -> deliver but flag for review
    min_margin: float = 0.08         # top-1 must lead runner-up by this, else review
    max_ood: float = 0.5             # fraction of hot spans unknown to baseline before escalate


@dataclass
class RoutedDecision:
    trace_id: str
    lane: Lane
    prediction: Prediction        # what the SRE receives (empty when ESCALATE)
    raw_prediction: Prediction    # the inner agent's verified output, before routing
    reasons: list[str] = field(default_factory=list)
    ood_score: float = 0.0
    groundedness: float = 1.0
    dropped_spans: list[str] = field(default_factory=list)  # hallucinated, removed by verification
    input_violations: list[str] = field(default_factory=list)


def ood_score(trace: Trace, baseline: LatencyBaseline, top: int = 5) -> float:
    """Fraction of the trace's hottest spans whose (service, operation) is unknown to the baseline.

    A grounded out-of-distribution signal (Kadavath: self-knowledge degrades OOD, so confidence is
    untrustworthy there). Computed from the spans that actually drive the diagnosis — the hottest
    by self-time — not the whole trace, so a few unfamiliar leaf calls don't trip it.
    """
    known = set(baseline.keys())
    hot = survey(trace, top=top).hottest_by_self_time
    if not hot:
        return 0.0
    unknown = sum(1 for v in hot if (v.service, v.operation) not in known)
    return unknown / len(hot)


def route(
    prediction: Prediction,
    trace: Trace,
    baseline: LatencyBaseline,
    thresholds: RouterThresholds,
) -> RoutedDecision:
    """Assign a lane to an already-verified prediction using the CP6 risk signals."""
    reasons: list[str] = []
    ood = ood_score(trace, baseline)
    grounded = groundedness(prediction, trace)

    # No surviving hypothesis is the agent's own "inconclusive within budget" -> escalate.
    if not prediction.ranked:
        reasons.append("agent returned inconclusive within budget")
        return RoutedDecision(trace.trace_id, Lane.ESCALATE, Prediction(trace.trace_id, []),
                              prediction, reasons, ood, grounded)

    top = prediction.ranked[0]
    runner = prediction.ranked[1].confidence if len(prediction.ranked) > 1 else 0.0
    margin = top.confidence - runner

    escalate = False
    review = False
    if ood > thresholds.max_ood:
        reasons.append(f"out-of-distribution trace (ood={ood:.2f}); calibration unreliable")
        escalate = True
    if top.confidence < thresholds.confidence_floor:
        reasons.append(f"top-1 confidence {top.confidence:.2f} below floor {thresholds.confidence_floor:.2f}")
        escalate = True

    if not escalate:
        if top.confidence < thresholds.review_ceiling:
            reasons.append(f"confidence {top.confidence:.2f} in review band")
            review = True
        if margin < thresholds.min_margin:
            reasons.append(f"thin margin {margin:.2f} to runner-up; ambiguous")
            review = True

    if escalate:
        lane, delivered = Lane.ESCALATE, Prediction(trace.trace_id, [])
    elif review:
        lane, delivered = Lane.REVIEW, prediction
    else:
        lane, delivered = Lane.AUTO, prediction
        reasons.append("confident, in-distribution, grounded")

    return RoutedDecision(trace.trace_id, lane, delivered, prediction, reasons, ood, grounded)


class SafeLocalizer:
    """Wrap any Localizer (Condition A/B/C) in the full CP6 control system.

    Itself a Localizer — `safe(trace) -> Prediction` — so it scores on the existing harness, where
    an ESCALATE shows up as an abstention (empty prediction) in escalation_rate and on the
    accuracy-coverage curve. Use `.decide(trace)` to get the full RoutedDecision (lane, reasons,
    OOD, groundedness, dropped hallucinations) for audit and the safety metrics.
    """

    def __init__(
        self,
        inner,
        baseline: LatencyBaseline,
        thresholds: RouterThresholds | None = None,
        max_spans: int = 2000,
    ) -> None:
        self._inner = inner
        self._baseline = baseline
        self._thresholds = thresholds or RouterThresholds()
        self._max_spans = max_spans

    def decide(self, trace: Trace) -> RoutedDecision:
        # 1. Input guardrails. A violation halts before the agent ever sees the trace.
        report = check_input(trace, max_spans=self._max_spans)
        if not report.ok:
            empty = Prediction(trace.trace_id, [])
            return RoutedDecision(trace.trace_id, Lane.ESCALATE, empty, empty,
                                  reasons=["input guardrail tripped: " + "; ".join(report.violations)],
                                  input_violations=report.violations)
        if report.redactions:
            redact_secrets(trace)

        # 2. Run the inner agent (Condition A/B/C).
        raw = self._inner(trace)

        # 3. Source-verify + schema-enforce: drop hallucinated spans before they can be ranked.
        verified, dropped = verify_prediction(raw, trace)

        # 4. Trust/risk router.
        decision = route(verified, trace, self._baseline, self._thresholds)
        decision.dropped_spans = dropped
        if dropped:
            decision.reasons.append(f"dropped {len(dropped)} hallucinated span(s)")
        return decision

    def __call__(self, trace: Trace) -> Prediction:
        return self.decide(trace).prediction
