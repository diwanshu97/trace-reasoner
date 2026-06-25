"""Run the A/B/C iso-budget comparison offline and print one table (the central experiment).

    .venv/bin/python eval_conditions.py            # conditions A/B/C on the mock
    .venv/bin/python eval_conditions.py --safe      # each condition under the CP6 safety system

No API key required: each condition runs on its deterministic offline mock, so this is the
reproducible comparison that the live runs (run_agent / run_multiagent / run_tot) mirror on Claude.
"""

import sys

from trace_reasoner.datasets.synthetic import SyntheticDataset
from trace_reasoner.eval.compare import build_conditions, comparison_table, compare


def main() -> None:
    safe = "--safe" in sys.argv
    dataset = SyntheticDataset(n=40, seed=7, error_ratio=0.0)
    conditions = build_conditions(safe=safe)
    reports = compare(dataset, conditions)

    title = "A/B/C comparison" + (" (under CP6 safety system)" if safe else "")
    print(f"{title} — {dataset.name}, n={len(dataset)}\n")
    print(comparison_table(reports))
    print("\nLegend: top-1/top-3 RCA accuracy, localization F1, Brier + ECE (calibration), escalation rate.")
    if safe:
        print("Under --safe, escalation rate counts traces the router sent to a human (abstentions).")


if __name__ == "__main__":
    main()
