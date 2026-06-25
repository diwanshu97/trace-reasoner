"""The scoring harness: run any localizer over a Dataset and report metrics.

A localizer is any Callable[[Trace], Prediction]. The vanilla ReAct baseline,
the multi-agent system, and the +ToT variant (the three ablation conditions)
are all just localizers, so they are all measured here on the same ruler.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from trace_reasoner.datasets.base import Dataset
from trace_reasoner.eval.metrics import (
    Prediction,
    brier_score,
    escalation_rate,
    expected_calibration_error,
    localization_f1,
    selective_accuracy_coverage,
    top_k_accuracy,
)
from trace_reasoner.trace import Trace

Localizer = Callable[[Trace], Prediction]


@dataclass
class EvalReport:
    dataset: str
    localizer: str
    n: int
    top_k: dict[int, float]
    localization_f1: float
    brier: float
    ece: float
    escalation: float
    coverage_curve: list[tuple[float, float, float]]

    def __str__(self) -> str:
        lines = [f"Eval: {self.localizer} on {self.dataset} (n={self.n})"]
        for k in sorted(self.top_k):
            lines.append(f"  top-{k} RCA accuracy : {self.top_k[k]:.3f}")
        lines.append(f"  localization F1     : {self.localization_f1:.3f}")
        lines.append(f"  Brier (top-1 calib) : {self.brier:.3f}")
        lines.append(f"  ECE (10-bin)        : {self.ece:.3f}")
        lines.append(f"  escalation rate     : {self.escalation:.3f}")
        lines.append("  accuracy-coverage   : " + ", ".join(
            f"@{t:.2f}->cov {cov:.2f}/acc {acc:.2f}" for t, cov, acc in self.coverage_curve
        ))
        return "\n".join(lines)


def evaluate(
    localizer: Localizer,
    dataset: Dataset,
    ks: tuple[int, ...] = (1, 3),
    name: str | None = None,
) -> EvalReport:
    pairs = [(localizer(ex.trace), ex.ground_truth) for ex in dataset]
    return EvalReport(
        dataset=getattr(dataset, "name", "dataset"),
        localizer=name or getattr(localizer, "__name__", "localizer"),
        n=len(pairs),
        top_k={k: top_k_accuracy(pairs, k) for k in ks},
        localization_f1=localization_f1(pairs),
        brier=brier_score(pairs),
        ece=expected_calibration_error(pairs),
        escalation=escalation_rate(pairs),
        coverage_curve=selective_accuracy_coverage(pairs),
    )
