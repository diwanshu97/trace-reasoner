"""Aggregate the A/B/C iso-budget comparison across seeds and fault mixes (the reproducible headline).

`eval_conditions.py` reports one seed; a single n=40 draw is not a defensible headline number.
This runs the same three conditions over several seeds and both fault mixes (pure-latency and
mixed latency/error), then reports the mean and spread per condition. Fully deterministic and
offline — no API key — so the table reproduces exactly on any machine.

    .venv/bin/python eval_aggregate.py

Read alongside eval_baseline.py (the slowest-leaf floor the agent must beat) and run_local.py /
run_tot.py (the live-model runs on the same ruler). The story this table tells: on the offline
mock, all three conditions sit near the accuracy ceiling on latency faults and converge on the
same answer on error faults, so architecture barely moves *accuracy* here — the robust, seed-stable
separation is in *calibration* (ECE), where the multi-agent specialists (B) are best and
Tree-of-Thought (C) is overconfident. Moving accuracy is a live-LLM effect; see the live runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from trace_reasoner.datasets.synthetic import SyntheticDataset
from trace_reasoner.eval.compare import CONDITION_LABELS, build_conditions, compare

SEEDS = (7, 11, 23, 42, 101)
FAULT_MIXES = {"latency": 0.0, "mixed": 0.3}
N_PER_RUN = 40


@dataclass
class Aggregate:
    """Mean and population stdev of one metric for one condition across all runs."""

    values: list[float]

    @property
    def mean(self) -> float:
        return mean(self.values) if self.values else 0.0

    @property
    def sd(self) -> float:
        return pstdev(self.values) if len(self.values) > 1 else 0.0

    def __str__(self) -> str:
        return f"{self.mean:.3f}±{self.sd:.3f}"


def aggregate_over(seeds=SEEDS, error_ratio: float = 0.0) -> dict[str, dict[str, Aggregate]]:
    """Run A/B/C over every seed at one fault mix; return {condition: {metric: Aggregate}}."""
    collected: dict[str, dict[str, list[float]]] = {
        key: {"top1": [], "top3": [], "f1": [], "ece": [], "brier": []} for key in ("A", "B", "C")
    }
    for seed in seeds:
        dataset = SyntheticDataset(n=N_PER_RUN, seed=seed, error_ratio=error_ratio)
        reports = compare(dataset, build_conditions())
        for key in ("A", "B", "C"):
            r = reports[key]
            collected[key]["top1"].append(r.top_k.get(1, 0.0))
            collected[key]["top3"].append(r.top_k.get(3, 0.0))
            collected[key]["f1"].append(r.localization_f1)
            collected[key]["ece"].append(r.ece)
            collected[key]["brier"].append(r.brier)
    return {
        key: {metric: Aggregate(vals) for metric, vals in metrics.items()}
        for key, metrics in collected.items()
    }


def render(title: str, agg: dict[str, dict[str, Aggregate]]) -> str:
    header = f"{'cond':<5} {'top-1':>12} {'top-3':>12} {'F1':>12} {'ECE':>12} {'Brier':>12}"
    lines = [title, header, "-" * len(header)]
    for key in ("A", "B", "C"):
        m = agg[key]
        lines.append(
            f"{key:<5} {str(m['top1']):>12} {str(m['top3']):>12} "
            f"{str(m['f1']):>12} {str(m['ece']):>12} {str(m['brier']):>12}"
        )
    return "\n".join(lines)


def main() -> None:
    print(f"A/B/C aggregate over seeds {SEEDS}, n={N_PER_RUN}/run (mean±sd, offline mock)\n")
    for key, label in CONDITION_LABELS.items():
        print(f"  {label}")
    print()
    for name, er in FAULT_MIXES.items():
        agg = aggregate_over(error_ratio=er)
        print(render(f"[{name} faults, error_ratio={er}]", agg))
        print()
    print("Legend: top-1/top-3 RCA accuracy, localization F1, ECE + Brier (calibration). Lower ECE/Brier"
          " is better. Compare top-1 against the slowest-leaf floor (eval_baseline.py): synthetic 0.39.")


if __name__ == "__main__":
    main()
