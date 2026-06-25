import pytest

from trace_reasoner.trace import Span, Trace


def make_trace() -> Trace:
    # r[0..100] ; a[10..40] and b[40..90] are children of r ; c[45..85] child of b
    spans = [
        Span("r", None, "gw", "GET", 0, 100),
        Span("a", "r", "svc", "rpc", 10, 30),
        Span("b", "r", "svc", "rpc", 40, 50),
        Span("c", "b", "db", "query", 45, 40),
    ]
    return Trace("t1", spans)


def test_root_and_children():
    t = make_trace()
    assert t.root.span_id == "r"
    assert {s.span_id for s in t.children("r")} == {"a", "b"}
    assert [s.span_id for s in t.children("b")] == ["c"]


def test_parent_and_ancestors():
    t = make_trace()
    assert t.parent("c").span_id == "b"
    assert [s.span_id for s in t.ancestors("c")] == ["b", "r"]
    assert t.parent("r") is None


def test_descendants_and_leaves():
    t = make_trace()
    assert {s.span_id for s in t.descendants("r")} == {"a", "b", "c"}
    assert {s.span_id for s in t.leaves()} == {"a", "c"}
    assert t.is_leaf("a") and not t.is_leaf("b")


def test_self_time_subtracts_child_intervals():
    t = make_trace()
    # r: children cover [10..40] + [40..90] = 80 of 100 -> self 20
    assert t.self_time_ms("r") == pytest.approx(20.0)
    # b: child c covers [45..85] = 40 of 50 -> self 10
    assert t.self_time_ms("b") == pytest.approx(10.0)
    # leaves: self == duration
    assert t.self_time_ms("a") == pytest.approx(30.0)


def test_critical_path_follows_latest_finisher():
    t = make_trace()
    # children of r: a ends 40, b ends 90 -> descend b -> c
    assert [s.span_id for s in t.critical_path()] == ["r", "b", "c"]


def test_end_to_end_duration():
    assert make_trace().duration_ms == 100


def test_rejects_two_roots():
    with pytest.raises(ValueError):
        Trace("t", [Span("a", None, "s", "o", 0, 1), Span("b", None, "s", "o", 0, 1)])


def test_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        Trace("t", [Span("x", None, "s", "o", 0, 1), Span("x", "x", "s", "o", 0, 1)])
