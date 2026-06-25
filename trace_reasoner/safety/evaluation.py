"""Safety-layer metrics: scoring the router's decisions (Checkpoint 6).

The harness metrics (eval/metrics.py) score the Prediction a SafeLocalizer delivers, so ECE,
accuracy-coverage, and escalation_rate already work on it unchanged. These extra metrics need the
full RoutedDecision — the lane and the reasons — so they live here, beside the router.

  escalation_precision  of the traces the system escalated, the fraction it SHOULD have (the agent
                        would have been wrong if it had answered autonomously). High precision means
                        escalations are genuinely hard cases, not noise that trains the SRE to ignore.
  autonomy_rate         fraction handled with no human (AUTO lane) — the efficiency side of the
                        autonomy/oversight trade-off.
  lane_distribution     counts per lane, for the safety dashboard.

`safety_report` rolls these into one object with the same __str__ style as EvalReport.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from trace_reasoner.datasets.base import GroundTruth
from trace_reasoner.safety.router import Lane, RoutedDecision

# A decision paired with the trace's ground truth — what the safety metrics consume.
ScoredDecision = tuple[RoutedDecision, GroundTruth]


def _would_have_been_wrong(decision: RoutedDecision, gt: GroundTruth) -> bool:
    """Would the agent's raw (pre-routing) top-1 have missed the true root cause?

    This is the counterfactual the escalation is judged against: an escalation is *correct* when
    answering autonomously would have been wrong (or there was nothing to answer).
    """
    if not decision.raw_prediction.ranked:
        return True  # nothing to offer -> escalating was right
    top = decision.raw_prediction.ranked[0].span_id
    return top not in set(gt.root_cause_span_ids)


def escalation_precision(scored: list[ScoredDecision]) -> float:
    """Of escalated traces, the fraction where autonomous answering would have been wrong.

    1.0 means every escalation was a genuinely hard case. Returns 1.0 when nothing was escalated
    (vacuously precise — read it alongside escalation_rate, which would be 0).
    """
    escalated = [(d, gt) for d, gt in scored if d.lane is Lane.ESCALATE]
    if not escalated:
        return 1.0
    correct = sum(1 for d, gt in escalated if _would_have_been_wrong(d, gt))
    return correct / len(escalated)


def autonomy_rate(scored: list[ScoredDecision]) -> float:
    """Fraction of traces handled fully autonomously (AUTO lane)."""
    if not scored:
        return 0.0
    return sum(1 for d, _ in scored if d.lane is Lane.AUTO) / len(scored)


def lane_distribution(scored: list[ScoredDecision]) -> dict[str, int]:
    counts = Counter(d.lane.value for d, _ in scored)
    return {lane.value: counts.get(lane.value, 0) for lane in Lane}


def mean_groundedness(scored: list[ScoredDecision]) -> float:
    if not scored:
        return 1.0
    return sum(d.groundedness for d, _ in scored) / len(scored)


@dataclass
class SafetyReport:
    n: int
    lanes: dict[str, int] = field(default_factory=dict)
    autonomy: float = 0.0
    escalation_precision: float = 1.0
    mean_groundedness: float = 1.0
    total_dropped_hallucinations: int = 0

    def __str__(self) -> str:
        lanes = ", ".join(f"{k} {v}" for k, v in self.lanes.items())
        return "\n".join([
            f"Safety report (n={self.n})",
            f"  lanes               : {lanes}",
            f"  autonomy rate       : {self.autonomy:.3f}",
            f"  escalation precision: {self.escalation_precision:.3f}",
            f"  mean groundedness   : {self.mean_groundedness:.3f}",
            f"  dropped hallucinations: {self.total_dropped_hallucinations}",
        ])


def safety_report(scored: list[ScoredDecision]) -> SafetyReport:
    return SafetyReport(
        n=len(scored),
        lanes=lane_distribution(scored),
        autonomy=autonomy_rate(scored),
        escalation_precision=escalation_precision(scored),
        mean_groundedness=mean_groundedness(scored),
        total_dropped_hallucinations=sum(len(d.dropped_spans) for d, _ in scored),
    )
