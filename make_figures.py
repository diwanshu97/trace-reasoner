"""Render the report figures from the same eval harness (reproducible, offline, no API key).

Produces two PNGs under figures/ that back the Final Report's evaluation section:

  figures/ece_by_condition.png    — calibration (ECE) per condition, both fault mixes; lower is better
  figures/accuracy_coverage.png   — accuracy-coverage curve on mixed faults; the selective-prediction view

Both average over the same 5 seeds as eval_aggregate.py, so the numbers match the report tables.

    .venv/bin/python make_figures.py
"""

from __future__ import annotations

from statistics import mean

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt

from trace_reasoner.datasets.synthetic import SyntheticDataset
from trace_reasoner.eval.compare import build_conditions, compare

SEEDS = (7, 11, 23, 42, 101)
N = 40
CONDS = ("A", "B", "C")
COND_COLORS = {"A": "#c0392b", "B": "#27ae60", "C": "#2980b9"}
COND_LABELS = {"A": "A: ReAct", "B": "B: multi-agent", "C": "C: +ToT"}
OUT = "figures"


def mean_ece(error_ratio: float) -> dict[str, float]:
    out = {}
    for key in CONDS:
        eces = [
            compare(SyntheticDataset(n=N, seed=s, error_ratio=error_ratio), build_conditions())[key].ece
            for s in SEEDS
        ]
        out[key] = mean(eces)
    return out


def mean_coverage_curve(error_ratio: float) -> dict[str, list[tuple[float, float, float]]]:
    ths = (0.0, 0.25, 0.5, 0.75, 0.9)
    out: dict[str, list[tuple[float, float, float]]] = {}
    for key in CONDS:
        covs = {t: [] for t in ths}
        accs = {t: [] for t in ths}
        for s in SEEDS:
            reps = compare(SyntheticDataset(n=N, seed=s, error_ratio=error_ratio), build_conditions())
            for t, cov, acc in reps[key].coverage_curve:
                covs[t].append(cov)
                accs[t].append(acc)
        out[key] = [(t, mean(covs[t]), mean(accs[t])) for t in ths]
    return out


def fig_ece() -> None:
    lat = mean_ece(0.0)
    mix = mean_ece(0.3)
    x = range(len(CONDS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - w / 2 for i in x], [lat[k] for k in CONDS], w, label="latency faults", color="#95a5a6")
    ax.bar([i + w / 2 for i in x], [mix[k] for k in CONDS], w, label="mixed faults", color="#34495e")
    ax.set_xticks(list(x))
    ax.set_xticklabels([COND_LABELS[k] for k in CONDS])
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("Calibration by condition (lower is better, mean over 5 seeds)")
    ax.legend()
    for i, k in enumerate(CONDS):
        ax.text(i - w / 2, lat[k] + 0.008, f"{lat[k]:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, mix[k] + 0.008, f"{mix[k]:.2f}", ha="center", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ece_by_condition.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/ece_by_condition.png")


def fig_coverage() -> None:
    curves = mean_coverage_curve(0.3)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key in CONDS:
        # Drop zero/near-zero-coverage thresholds: accuracy is undefined when nothing is answered.
        pts = [r for r in curves[key] if r[1] >= 0.02]
        pts = sorted(pts, key=lambda r: r[1])  # sort by coverage for a readable line
        cov = [c for _, c, _ in pts]
        acc = [a for _, _, a in pts]
        ax.plot(cov, acc, "o-", color=COND_COLORS[key], label=COND_LABELS[key], linewidth=2, markersize=7)
    ax.set_xlabel("Coverage (fraction of traces answered)")
    ax.set_ylabel("Accuracy on answered traces")
    ax.set_title("Accuracy-coverage on mixed faults (up-and-right is better)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.5, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.annotate("A answers everything,\nwrong on error faults", xy=(1.0, 0.69), xytext=(0.62, 0.62),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.annotate("B abstains on what it\ncannot localize", xy=(0.70, 0.97), xytext=(0.30, 0.85),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#27ae60"))
    fig.tight_layout()
    fig.savefig(f"{OUT}/accuracy_coverage.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/accuracy_coverage.png")


def main() -> None:
    import os

    os.makedirs(OUT, exist_ok=True)
    fig_ece()
    fig_coverage()


if __name__ == "__main__":
    main()
