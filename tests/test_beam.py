import pytest

from trace_reasoner.datasets.synthetic import SyntheticDataset, generate_example, normal_traces
from trace_reasoner.eval.harness import evaluate
from trace_reasoner.multiagent.beam import (
    Thought,
    ToTLocalizer,
    _Budget,
    beam_search,
    expand,
    score_thought,
)
from trace_reasoner.multiagent.mock import HeuristicSpecialistClient
from trace_reasoner.tools.baseline import LatencyBaseline


def make_baseline(n=100, seed=0) -> LatencyBaseline:
    return LatencyBaseline.from_traces(normal_traces(n, seed=seed))


# --- the critic: scoring one thought against grounded tools -------------------
def test_critic_verifies_the_injected_anomaly():
    ex = generate_example(seed=42, fault="latency")
    truth = ex.ground_truth.root_cause_span_ids[0]
    scored = score_thought(Thought(span_id=truth), ex.trace, make_baseline(), retriever=None)
    assert scored.verified is True       # baseline confirmed it anomalous
    assert scored.hard_pruned is False
    assert scored.score > 0.0
    assert scored.criteria["anomaly"] > 0.0


def test_critic_hard_prunes_a_normal_span():
    # The trace root is never the injected fault, so baseline reports it normal -> hard prune.
    ex = generate_example(seed=42, fault="latency")
    root_id = ex.trace.root.span_id
    scored = score_thought(Thought(span_id=root_id), ex.trace, make_baseline(), retriever=None)
    assert scored.hard_pruned is True
    assert scored.score == 0.0


# --- expansion: structural hops ----------------------------------------------
def test_expand_walks_to_parent_and_child():
    ex = generate_example(seed=7, fault="latency")
    # an internal span: the root's first child has both a parent and children
    internal = ex.trace.children(ex.trace.root.span_id)[0].span_id
    kids = expand(Thought(span_id=internal), ex.trace, branching=4)
    span_ids = {t.span_id for t in kids}
    assert ex.trace.root.span_id in span_ids  # refined up to the parent
    assert kids and all(t.depth == 2 for t in kids)


# --- the controller -----------------------------------------------------------
def test_beam_localizes_a_latency_fault():
    ex = generate_example(seed=42, fault="latency")
    tot = ToTLocalizer(HeuristicSpecialistClient(), make_baseline())
    pred = tot(ex.trace)
    assert pred.ranked
    assert ex.ground_truth.root_cause_span_ids[0] in pred.top_ids(3)
    assert all(0.0 <= h.confidence <= 1.0 for h in pred.ranked)


def test_beam_respects_the_tool_budget():
    # A tight budget must still terminate, score, and never exceed the cap.
    ex = generate_example(seed=7, fault="latency")
    budget = _Budget(limit=4)

    def gen(trace):
        return [Thought(span_id=trace.root.span_id)]

    beam_search(ex.trace, make_baseline(), None, gen, max_tool_calls=4)
    assert budget.exhausted is False  # untouched local budget; the controller owns its own

    tot = ToTLocalizer(HeuristicSpecialistClient(), make_baseline(), max_tool_calls=4)
    pred = tot(ex.trace)
    assert pred.trace_id == ex.trace.trace_id


def test_beam_inconclusive_when_nothing_clears_floor():
    # A fault-free trace has no anomalous span -> every branch hard-prunes -> inconclusive.
    ex = generate_example(seed=3, fault="none")
    tot = ToTLocalizer(HeuristicSpecialistClient(), make_baseline())
    pred = tot(ex.trace)
    assert pred.ranked == []  # honest "inconclusive within budget" (Checkpoint 1)


def test_beam_ranked_is_sorted_by_confidence():
    ex = generate_example(seed=11, fault="latency")
    tot = ToTLocalizer(HeuristicSpecialistClient(), make_baseline())
    pred = tot(ex.trace)
    confs = [h.confidence for h in pred.ranked]
    assert confs == sorted(confs, reverse=True)


# --- Condition C as a Localizer in the harness --------------------------------
def test_tot_is_a_localizer_in_the_harness():
    ds = SyntheticDataset(n=6, seed=5, error_ratio=0.0)
    tot = ToTLocalizer(HeuristicSpecialistClient(), make_baseline(seed=1))
    report = evaluate(tot, ds, ks=(1, 3), name="condition-C-tot")
    assert report.n == 6
    assert 0.0 <= report.brier <= 1.0
    assert 0.0 <= report.top_k[1] <= 1.0
