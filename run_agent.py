"""Live smoke test: run the ReAct agent on Claude over a few synthetic traces.

Requires the agent extra and an API key:
    .venv/bin/pip install -e '.[agent]'
    ANTHROPIC_API_KEY=sk-... .venv/bin/python run_agent.py

Defaults to a handful of synthetic traces so a live run stays cheap. Swap in
NezhaDataset to try real traces.
"""

from trace_reasoner.agent.anthropic_client import AnthropicClient
from trace_reasoner.agent.react import ReActLocalizer
from trace_reasoner.datasets.synthetic import SyntheticDataset, normal_traces
from trace_reasoner.eval.harness import evaluate
from trace_reasoner.tools.baseline import LatencyBaseline


def main() -> None:
    baseline = LatencyBaseline.from_traces(normal_traces(100, seed=0))
    dataset = SyntheticDataset(n=5, seed=99)  # small + cheap for a live smoke test
    agent = ReActLocalizer(AnthropicClient(model="claude-opus-4-8"), baseline, max_steps=15)

    report = evaluate(agent, dataset, ks=(1, 3), name="react-claude")
    print(report)


if __name__ == "__main__":
    main()
