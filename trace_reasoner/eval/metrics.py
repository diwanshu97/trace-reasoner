"""Prediction types and the metrics (Checkpoints 5 and 6).

A localizer outputs a Prediction: a ranked list of root-cause Hypotheses with
confidences. The metrics:

  top_k_accuracy        — did a true root-cause span land in the top k? (headline RCA)
  localization_f1       — set overlap between predicted and true root-cause spans
  brier_score           — calibration of the top-1 hypothesis's confidence

Checkpoint 6 adds the safety/calibration metrics — calibration graded over raw
accuracy is the commitment from Checkpoint 1, so these are first-class:

  expected_calibration_error  — ECE (HELM): does an 0.8 confidence mean ~80% right?
  selective_accuracy_coverage — accuracy vs the answered fraction (HELM selective prediction);
                                the instrument for the "inconclusive within budget" abstention
  escalation_rate             — fraction of traces the agent abstains on (empty prediction)

All take a list of (Prediction, GroundTruth) pairs so they share one path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from trace_reasoner.datasets.base import GroundTruth

# A scored pair the metrics consume.
Pair = tuple["Prediction", GroundTruth]


@dataclass
class Hypothesis:
    span_id: str
    confidence: float = 0.0  # 0..1
    fault_family: str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    """A localizer's ranked root-cause hypotheses for one trace.

    `ranked` must be sorted descending by confidence (the localizer's job).
    """

    trace_id: str
    ranked: list[Hypothesis]

    def top_ids(self, k: int) -> list[str]:
        return [h.span_id for h in self.ranked[:k]]


def top_k_accuracy(pairs: list[Pair], k: int) -> float:
    """Fraction of traces where a true root-cause span is in the top-k prediction."""
    if not pairs:
        return 0.0
    hits = 0
    for pred, gt in pairs:
        if set(pred.top_ids(k)) & set(gt.root_cause_span_ids):
            hits += 1
    return hits / len(pairs)


def localization_f1(pairs: list[Pair], k: int | None = None) -> float:
    """Mean per-trace F1 between predicted and true root-cause span sets.

    By default the predicted set is the top-`m` spans where m = number of true
    root-cause spans (so a single-span truth is a top-1 hit/miss, and a span
    chain is scored as set overlap). Pass `k` to fix the predicted-set size.
    """
    if not pairs:
        return 0.0
    f1s: list[float] = []
    for pred, gt in pairs:
        true = set(gt.root_cause_span_ids)
        if not true:
            continue
        m = k if k is not None else max(1, len(true))
        predicted = set(pred.top_ids(m))
        tp = len(predicted & true)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(true)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)
    return mean(f1s) if f1s else 0.0


def brier_score(pairs: list[Pair]) -> float:
    """Mean squared error of the top-1 hypothesis confidence vs its correctness.

    Lower is better. An empty prediction is treated as confidence 0 on an
    outcome of 0 (an honest abstention), contributing 0 error.
    """
    if not pairs:
        return 0.0
    errs: list[float] = []
    for pred, gt in pairs:
        if not pred.ranked:
            errs.append(0.0)
            continue
        top = pred.ranked[0]
        outcome = 1.0 if top.span_id in set(gt.root_cause_span_ids) else 0.0
        errs.append((top.confidence - outcome) ** 2)
    return mean(errs)


def _answered(pairs: list[Pair]) -> list[tuple[float, float]]:
    """The (confidence, correctness) of the top-1 hypothesis on traces the agent answered.

    Abstentions (empty predictions, the "inconclusive within budget" outcome) are excluded —
    they carry no confidence to calibrate and no answer to score. Both ECE and the
    accuracy-coverage curve operate on this answered set.
    """
    out: list[tuple[float, float]] = []
    for pred, gt in pairs:
        if not pred.ranked:
            continue
        top = pred.ranked[0]
        correct = 1.0 if top.span_id in set(gt.root_cause_span_ids) else 0.0
        out.append((min(1.0, max(0.0, top.confidence)), correct))
    return out


def expected_calibration_error(pairs: list[Pair], bins: int = 10) -> float:
    """ECE over the top-1 confidence (HELM, 10 equal-width bins by default).

    Bins the answered predictions by confidence, and for each bin takes
    |mean confidence - accuracy| weighted by the bin's share of answered traces. 0 is perfect
    calibration. Abstentions are excluded, so ECE measures the calibration of what the agent
    actually claimed — the Checkpoint 6 headline metric.
    """
    answered = _answered(pairs)
    if not answered:
        return 0.0
    total = len(answered)
    error = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        # last bin is closed on the right so confidence == 1.0 lands somewhere
        members = [(c, y) for c, y in answered if (lo <= c < hi or (b == bins - 1 and c == 1.0))]
        if not members:
            continue
        conf = mean(c for c, _ in members)
        acc = mean(y for _, y in members)
        error += (len(members) / total) * abs(conf - acc)
    return error


def selective_accuracy_coverage(pairs: list[Pair], thresholds: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9)) -> list[tuple[float, float, float]]:
    """Accuracy-coverage curve (HELM selective prediction).

    For each confidence threshold, report (threshold, coverage, accuracy) where coverage is the
    fraction of *all* traces whose top-1 confidence clears the threshold, and accuracy is the
    top-1 RCA rate on just those. A well-calibrated agent that abstains on its weak cases should
    show accuracy *rising* as coverage falls — the formal justification for "inconclusive within
    budget" (Checkpoint 1): refusing to answer the hard cases buys accuracy on the rest.
    """
    if not pairs:
        return [(t, 0.0, 0.0) for t in thresholds]
    answered = _answered(pairs)
    n = len(pairs)
    curve: list[tuple[float, float, float]] = []
    for t in thresholds:
        kept = [(c, y) for c, y in answered if c >= t]
        coverage = len(kept) / n
        accuracy = mean(y for _, y in kept) if kept else 0.0
        curve.append((t, round(coverage, 3), round(accuracy, 3)))
    return curve


def escalation_rate(pairs: list[Pair]) -> float:
    """Fraction of traces the agent abstained on (empty prediction → routed to a human).

    The Checkpoint 6 escalation signal: too high and the on-call SRE is over-paged until they
    ignore the tool; too low and confident-but-wrong leaks through. Read alongside the
    accuracy-coverage curve to choose the operating point.
    """
    if not pairs:
        return 0.0
    return sum(1 for pred, _ in pairs if not pred.ranked) / len(pairs)
