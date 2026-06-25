"""Live smoke test: run Condition C (Tree-of-Thought beam search) on Claude over a few traces.

Requires the agent + multiagent extras and an API key:
    .venv/bin/pip install -e '.[agent,multiagent,rag]'
    ANTHROPIC_API_KEY=sk-... .venv/bin/python run_tot.py

Condition C is Condition B with a beam search spliced into the hypothesize phase: the three
specialists (latency, dependency, pattern) run on Claude as thought generators, then a
deterministic critic + controller score, prune, and expand the hypotheses in plain Python. The
tool budget (beam width x depth) is the controlled variable, so this scores on the same harness
and the same iso-token ruler as Conditions A and B. Swap in NezhaDataset for real traces.
"""

from trace_reasoner.agent.anthropic_client import AnthropicClient
from trace_reasoner.datasets.synthetic import SyntheticDataset, normal_traces
from trace_reasoner.eval.harness import evaluate
from trace_reasoner.multiagent.beam import ToTLocalizer
from trace_reasoner.tools.baseline import LatencyBaseline


def main() -> None:
    baseline = LatencyBaseline.from_traces(normal_traces(100, seed=0))
    dataset = SyntheticDataset(n=5, seed=99)  # small + cheap for a live smoke test

    retriever = None
    try:
        from trace_reasoner.rag.retriever import PrecedentRetriever

        retriever = PrecedentRetriever.production()
    except Exception as exc:  # rag extra not installed, or model download blocked
        print(f"(critic running without retriever: {exc})")

    llm = AnthropicClient(model="claude-opus-4-8")
    tot = ToTLocalizer(llm, baseline, retriever=retriever, beam_width=3, max_depth=4)

    report = evaluate(tot, dataset, ks=(1, 3), name="condition-C-tot")
    print(report)


if __name__ == "__main__":
    main()
