"""Run the slowest-leaf baseline over real Nezha traces and print a report.

    .venv/bin/python eval_nezha.py

Requires the dataset cloned to data/Nezha/ (see README).
"""

from trace_reasoner.baselines.slowest_leaf import slowest_leaf
from trace_reasoner.datasets.nezha import NezhaDataset
from trace_reasoner.eval.harness import evaluate


def main() -> None:
    dataset = NezhaDataset(system="hipster", max_traces=200, max_traces_per_fault=10, seed=1)
    print(f"loaded {len(dataset)} labeled traces from Nezha ({dataset.system})")
    report = evaluate(slowest_leaf, dataset, ks=(1, 3), name="slowest_leaf")
    print(report)


if __name__ == "__main__":
    main()
