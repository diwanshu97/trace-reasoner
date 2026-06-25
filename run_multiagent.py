"""Live smoke test: run Condition B (the multi-agent system) on Claude over a few synthetic traces.

Requires the agent + multiagent extras and an API key:
    .venv/bin/pip install -e '.[agent,multiagent]'
    ANTHROPIC_API_KEY=sk-... .venv/bin/python run_multiagent.py

Each specialist (latency, dependency, pattern) runs its own bounded ReAct loop on Claude; the
synthesizer reconciles them in plain Python. The pattern analyst uses the BGE precedent retriever
if the rag extra is installed, and degrades gracefully without it. Swap in NezhaDataset for real traces.
"""

from trace_reasoner.agent.anthropic_client import AnthropicClient
from trace_reasoner.datasets.synthetic import SyntheticDataset, normal_traces
from trace_reasoner.eval.harness import evaluate
from trace_reasoner.multiagent.graph import MultiAgentLocalizer
from trace_reasoner.tools.baseline import LatencyBaseline


def main() -> None:
    baseline = LatencyBaseline.from_traces(normal_traces(100, seed=0))
    dataset = SyntheticDataset(n=5, seed=99)  # small + cheap for a live smoke test

    retriever = None
    try:
        from trace_reasoner.rag.retriever import PrecedentRetriever

        retriever = PrecedentRetriever.production()
    except Exception as exc:  # rag extra not installed, or model download blocked
        print(f"(pattern analyst running without retriever: {exc})")

    llm = AnthropicClient(model="claude-opus-4-8")
    mas = MultiAgentLocalizer(llm, baseline, retriever=retriever, max_rounds=2)

    report = evaluate(mas, dataset, ks=(1, 3), name="condition-B-multiagent")
    print(report)


if __name__ == "__main__":
    main()
