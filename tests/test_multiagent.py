import importlib.util

import pytest

from trace_reasoner.datasets.synthetic import SyntheticDataset, generate_example, normal_traces
from trace_reasoner.eval.harness import evaluate
from trace_reasoner.multiagent.mock import HeuristicSpecialistClient
from trace_reasoner.multiagent.specialists import run_specialist
from trace_reasoner.multiagent.state import Finding
from trace_reasoner.multiagent.synthesizer import reconcile, redispatch_hint, redispatch_targets
from trace_reasoner.tools.baseline import LatencyBaseline

langgraph_installed = importlib.util.find_spec("langgraph") is not None
requires_langgraph = pytest.mark.skipif(not langgraph_installed, reason="langgraph not installed")


def make_baseline(n=100, seed=0) -> LatencyBaseline:
    return LatencyBaseline.from_traces(normal_traces(n, seed=seed))


# --- specialists (no langgraph dependency) ------------------------------------
def test_latency_specialist_grounds_and_submits():
    ex = generate_example(seed=42, fault="latency")
    f = run_specialist("latency", HeuristicSpecialistClient(), ex.trace, make_baseline())
    assert f.role == "latency"
    assert f.span_id is not None
    assert f.anomalous is not None  # latency lens reports the grounded flag
    assert 0.0 <= f.confidence <= 1.0


def test_dependency_specialist_submits_a_structural_span():
    ex = generate_example(seed=7, fault="latency")
    f = run_specialist("dependency", HeuristicSpecialistClient(), ex.trace, make_baseline())
    assert f.role == "dependency"
    assert f.span_id is not None


def test_pattern_specialist_runs_without_a_retriever():
    # No retriever injected → the pattern lens must degrade gracefully, not crash.
    ex = generate_example(seed=7, fault="latency")
    f = run_specialist("pattern", HeuristicSpecialistClient(), ex.trace, make_baseline(), retriever=None)
    assert f.role == "pattern"
    assert 0.0 <= f.confidence <= 1.0


def test_unknown_role_raises():
    ex = generate_example(seed=1, fault="latency")
    with pytest.raises(ValueError):
        run_specialist("security", HeuristicSpecialistClient(), ex.trace, make_baseline())


# --- synthesizer (pure python) ------------------------------------------------
def test_reconcile_ranks_pooled_evidence_and_calibrates():
    findings = [
        Finding(role="latency", span_id="s1", confidence=0.85, anomalous=True, fault_family="saturation"),
        Finding(role="dependency", span_id="s1", confidence=0.6, fault_family="dependency"),
        Finding(role="pattern", span_id="s2", confidence=0.55, fault_family="saturation"),
    ]
    pred = reconcile("t1", findings)
    assert pred.ranked
    # s1 is named by two lenses incl. a verified latency hit → it should lead s2.
    assert pred.ranked[0].span_id == "s1"
    assert all(0.0 <= h.confidence <= 1.0 for h in pred.ranked)


def test_reconcile_empty_is_inconclusive():
    findings = [Finding(role="latency", span_id=None, confidence=0.0)]
    pred = reconcile("t1", findings)
    assert pred.ranked == []  # honest "inconclusive within budget"


def test_redispatch_fires_on_unverified_structural_lead():
    # Dependency confidently implicates s2; latency only verified s1 → re-check s2.
    findings = [
        Finding(role="latency", span_id="s1", confidence=0.85, anomalous=True),
        Finding(role="dependency", span_id="s2", confidence=0.7, fault_family="saturation"),
    ]
    assert redispatch_targets(findings) == ["latency"]
    assert "s2" in redispatch_hint(findings)


def test_no_redispatch_when_lenses_agree():
    findings = [
        Finding(role="latency", span_id="s1", confidence=0.85, anomalous=True),
        Finding(role="dependency", span_id="s1", confidence=0.7),
    ]
    assert redispatch_targets(findings) == []


# --- full graph as a Localizer ------------------------------------------------
@requires_langgraph
def test_mas_is_a_localizer_in_the_harness():
    from trace_reasoner.multiagent.graph import MultiAgentLocalizer

    ds = SyntheticDataset(n=8, seed=5, error_ratio=0.0)
    mas = MultiAgentLocalizer(HeuristicSpecialistClient(), make_baseline(seed=1))
    report = evaluate(mas, ds, ks=(1, 3), name="mas-mock")
    assert report.n == 8
    assert 0.0 <= report.brier <= 1.0
    assert 0.0 <= report.top_k[1] <= 1.0


@requires_langgraph
def test_mas_localizes_a_latency_fault():
    from trace_reasoner.multiagent.graph import MultiAgentLocalizer

    ex = generate_example(seed=42, fault="latency")
    mas = MultiAgentLocalizer(HeuristicSpecialistClient(), make_baseline())
    pred = mas(ex.trace)
    assert pred.ranked
    assert pred.ranked[0].span_id == ex.ground_truth.root_cause_span_ids[0]


@requires_langgraph
def test_mas_bounds_rounds():
    # max_rounds=1 means the synthesizer never re-dispatches; the graph still terminates and scores.
    from trace_reasoner.multiagent.graph import MultiAgentLocalizer

    ex = generate_example(seed=7, fault="latency")
    mas = MultiAgentLocalizer(HeuristicSpecialistClient(), make_baseline(), max_rounds=1)
    pred = mas(ex.trace)
    assert pred.trace_id == ex.trace.trace_id
