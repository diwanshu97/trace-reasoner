from trace_reasoner.agent.llm import AssistantTurn, ToolCall
from trace_reasoner.agent.mock import HeuristicMockClient
from trace_reasoner.agent.react import ReActLocalizer
from trace_reasoner.datasets.synthetic import SyntheticDataset, generate_example, normal_traces
from trace_reasoner.eval.harness import evaluate
from trace_reasoner.tools.baseline import LatencyBaseline


def make_baseline(n=100, seed=0) -> LatencyBaseline:
    return LatencyBaseline.from_traces(normal_traces(n, seed=seed))


def test_loop_localizes_a_latency_fault():
    agent = ReActLocalizer(HeuristicMockClient(), make_baseline(), max_steps=10)
    ex = generate_example(seed=42, fault="latency")
    pred = agent(ex.trace)
    assert pred.ranked
    # the mock follows survey -> hottest self-time -> submit; for a latency fault
    # the injected span is the hottest, so it should be the top hypothesis
    assert pred.ranked[0].span_id == ex.ground_truth.root_cause_span_ids[0]


def test_loop_beats_slowest_leaf_on_synthetic_latency():
    ds = SyntheticDataset(n=40, seed=3, error_ratio=0.0)  # latency-only
    agent = ReActLocalizer(HeuristicMockClient(), make_baseline(), max_steps=10)
    report = evaluate(agent, ds, ks=(1, 3), name="react-mock")
    # slowest_leaf manages ~0.39 top-1 on synthetic; the self-time agent clears it easily
    assert report.top_k[1] > 0.8


def test_loop_is_a_localizer_in_the_harness():
    ds = SyntheticDataset(n=10, seed=5, error_ratio=0.0)
    agent = ReActLocalizer(HeuristicMockClient(), make_baseline(seed=1))
    report = evaluate(agent, ds)
    assert report.n == 10
    assert 0.0 <= report.brier <= 1.0


def test_budget_exhaustion_falls_back_instead_of_crashing():
    class NeverSubmits:
        def respond(self, system, messages, tools):
            return AssistantTurn(text="looking...", tool_calls=[ToolCall("x", "survey", {})])

    agent = ReActLocalizer(NeverSubmits(), make_baseline(n=20), max_steps=3)
    pred = agent(generate_example(seed=1, fault="latency").trace)
    assert pred.ranked  # fallback still produces ranked hypotheses


def test_prose_only_turn_is_nudged_then_can_finish():
    # First a prose-only turn (no tool calls), then a submit — loop must survive the nudge.
    class ProseThenSubmit:
        def __init__(self):
            self.n = 0

        def respond(self, system, messages, tools):
            self.n += 1
            if self.n == 1:
                return AssistantTurn(text="I think it's the database.", tool_calls=[])
            return AssistantTurn(
                text="submitting",
                tool_calls=[ToolCall("s", "submit_hypotheses",
                                     {"hypotheses": [{"span_id": "syn-000007-s1", "confidence": 0.5}]})],
            )

    agent = ReActLocalizer(ProseThenSubmit(), make_baseline(n=20), max_steps=5)
    pred = agent(generate_example(seed=7, fault="latency").trace)
    assert pred.ranked and 0.0 <= pred.ranked[0].confidence <= 1.0
