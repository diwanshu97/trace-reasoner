"""Live run on a LOCAL open-weight model (Qwen2.5-7B / Llama-3.1-8B via Ollama) — the free backend.

The model-capability axis of the capstone: run the full A/B/C ablation on a small open model an
order of magnitude smaller than Claude, and see whether the architecture gains survive. Same
harness, same iso-budget ruler as the Claude runs — only the brain changes.

    # one-time setup, in another terminal:
    brew install ollama && ollama serve
    ollama pull qwen2.5:7b-instruct
    # then:
    .venv/bin/pip install -e '.[agent,multiagent,ui]'   # 'agent' pulls openai; rag optional
    .venv/bin/python run_local.py                       # A/B/C on the local model, tiny set
    .venv/bin/python run_local.py --safe --n 3 --model llama3.1:8b-instruct-q4_0

Honest expectations: 7-8B models are slow on a laptop (minutes per trace once B/C fan out into
specialists) and flakier at tool calling than Claude, so the default trace set is tiny and results
are noisier. That gap versus Claude is the finding, not a bug.
"""

import argparse

from trace_reasoner.agent.local_client import DEFAULT_MODEL, LocalClient
from trace_reasoner.datasets.synthetic import SyntheticDataset
from trace_reasoner.eval.compare import build_conditions, comparison_table, compare


def main() -> None:
    ap = argparse.ArgumentParser(description="Run A/B/C on a local open-weight model.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama/OpenAI-compatible model tag")
    ap.add_argument("--base-url", default=None, help="override the OpenAI-compatible endpoint")
    ap.add_argument("--n", type=int, default=3, help="number of traces (keep small; local is slow)")
    ap.add_argument("--error-ratio", type=float, default=0.0,
                    help="fraction of error (vs latency) faults; 0.3 mixes in the hard error cases")
    ap.add_argument("--safe", action="store_true", help="wrap each condition in the CP6 safety system")
    args = ap.parse_args()

    kwargs = {"model": args.model}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    llm = LocalClient(**kwargs)

    dataset = SyntheticDataset(n=args.n, seed=7, error_ratio=args.error_ratio)
    conditions = build_conditions(live_llm=llm, safe=args.safe)

    print(f"Running A/B/C on local model '{args.model}' — {dataset.name}, n={len(dataset)}")
    print("(this is slow on a laptop; B and C fan out into specialist sub-agents per trace)\n")
    reports = compare(dataset, conditions)
    print(comparison_table(reports))
    print("\nLegend: top-1/top-3 RCA accuracy, localization F1, Brier + ECE (calibration), escalation.")
    print("Compare against eval_conditions.py (mock) and run_tot.py (Claude) on the same ruler.")


if __name__ == "__main__":
    main()
