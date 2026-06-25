import pytest

from trace_reasoner.datasets.nezha import DEFAULT_ROOT
from trace_reasoner.datasets.synthetic import generate_example, normal_traces
from trace_reasoner.tools.baseline import LatencyBaseline, baseline_latency
from trace_reasoner.tools.walk_tree import survey, view, walk
from trace_reasoner.trace import Span, Trace

needs_data = pytest.mark.skipif(
    not (DEFAULT_ROOT / "construct_data").exists(),
    reason="Nezha dataset not cloned to data/Nezha",
)


def small() -> Trace:
    return Trace(
        "t",
        [
            Span("r", None, "gw", "GET", 0, 100),
            Span("a", "r", "svc", "rpc", 10, 30),
            Span("b", "r", "db", "query", 40, 50),
        ],
    )


# --- walk_tree -----------------------------------------------------------
def test_view_and_walk_directions():
    t = small()
    assert view(t, "r").n_children == 2
    assert {v.span_id for v in walk(t, "r", "children")} == {"a", "b"}
    assert walk(t, "a", "parent")[0].span_id == "r"
    assert [v.span_id for v in walk(t, "b", "siblings")] == ["a"]
    assert [v.span_id for v in walk(t, "a", "ancestors")] == ["r"]
    assert walk(t, "r", "parent") == []  # root has no parent


def test_walk_rejects_bad_direction():
    with pytest.raises(ValueError):
        walk(small(), "r", "sideways")


def test_survey_surfaces_structure():
    t = small()
    sv = survey(t)
    assert sv.n_spans == 3
    assert sv.duration_ms == 100
    assert set(sv.services) == {"gw", "svc", "db"}
    assert sv.critical_path[0].span_id == "r"  # path starts at root
    assert sv.hottest_by_self_time[0].self_time_ms >= sv.hottest_by_self_time[-1].self_time_ms


def test_survey_counts_error_spans():
    ex = generate_example(seed=5, fault="error")
    assert survey(ex.trace).error_spans >= 1


# --- baseline_latency ----------------------------------------------------
def test_baseline_flags_injected_latency():
    base = LatencyBaseline.from_traces(normal_traces(80, seed=0))
    assert len(base) > 0
    ex = generate_example(seed=123, fault="latency")
    target = ex.ground_truth.root_cause_span_ids[0]
    v = baseline_latency(base, ex.trace, target)
    if v.known:  # the injected span's (service, op) was seen in normals
        assert v.is_anomalous
        assert v.observed_ms > v.p95


def test_baseline_spares_normal_spans():
    base = LatencyBaseline.from_traces(normal_traces(80, seed=1))
    ex = generate_example(seed=7, fault="none")
    known = flagged = 0
    for s in ex.trace.spans:
        v = baseline_latency(base, ex.trace, s.span_id)
        if v.known:
            known += 1
            flagged += int(v.is_anomalous)
    assert known > 0
    assert flagged <= known * 0.5  # a fault-free trace should rarely trip


def test_baseline_unknown_op_is_not_anomalous():
    base = LatencyBaseline.from_traces(normal_traces(20, seed=2))
    v = base.verdict("never-seen-svc", "never-seen-op", 9_999.0)
    assert not v.known
    assert not v.is_anomalous


@needs_data
def test_baseline_builds_from_real_nezha_normals():
    base = LatencyBaseline.from_nezha(system="hipster", max_files=2)
    assert len(base) > 0
    service, operation = base.keys()[0]
    v = base.verdict(service, operation, 10_000_000.0)  # absurd latency
    assert v.known and v.is_anomalous
