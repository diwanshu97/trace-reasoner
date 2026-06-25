"""Run the slowest-leaf baseline over synthetic traces and print a report.

    .venv/bin/python eval_baseline.py
"""

from trace_reasoner.baselines.slowest_leaf import slowest_leaf
from trace_reasoner.datasets.synthetic import SyntheticDataset
from trace_reasoner.eval.harness import evaluate


def main() -> None:
    dataset = SyntheticDataset(n=200, seed=1)
    report = evaluate(slowest_leaf, dataset, ks=(1, 3), name="slowest_leaf")
    print(report)


if __name__ == "__main__":
    main()
