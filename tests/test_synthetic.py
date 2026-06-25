from statistics import median

from trace_reasoner.datasets.synthetic import SyntheticDataset, generate_example


def test_ground_truth_span_exists_in_trace():
    ex = generate_example(seed=1, fault="latency")
    ids = {s.span_id for s in ex.trace.spans}
    assert ex.ground_truth.root_cause_span_ids
    assert set(ex.ground_truth.root_cause_span_ids) <= ids


def test_deterministic_for_same_seed():
    a = generate_example(seed=7)
    b = generate_example(seed=7)
    assert [s.span_id for s in a.trace.spans] == [s.span_id for s in b.trace.spans]
    assert [s.duration_ms for s in a.trace.spans] == [s.duration_ms for s in b.trace.spans]
    assert a.ground_truth.root_cause_span_ids == b.ground_truth.root_cause_span_ids


def test_latency_fault_inflates_target_self_time():
    ex = generate_example(seed=3, fault="latency")
    t = ex.trace
    target = ex.ground_truth.root_cause_span_ids[0]
    self_times = [t.self_time_ms(s.span_id) for s in t.spans]
    # injected self-time should dominate the typical span
    assert t.self_time_ms(target) > 3 * median(self_times)
    assert ex.ground_truth.fault_family in {"saturation", "contention"}


def test_error_fault_marks_target_and_ancestors():
    ex = generate_example(seed=5, fault="error")
    t = ex.trace
    target = ex.ground_truth.root_cause_span_ids[0]
    assert t.get(target).is_error
    assert all(a.is_error for a in t.ancestors(target))
    assert ex.ground_truth.fault_family in {"exception", "dependency"}


def test_dataset_length_and_labels():
    ds = SyntheticDataset(n=12, seed=0)
    examples = list(ds)
    assert len(examples) == 12 == len(ds)
    for ex in examples:
        ids = {s.span_id for s in ex.trace.spans}
        assert set(ex.ground_truth.root_cause_span_ids) <= ids


def test_dataset_is_reiterable_and_deterministic():
    ds = SyntheticDataset(n=8, seed=2)
    first = [ex.trace.trace_id for ex in ds]
    second = [ex.trace.trace_id for ex in ds]
    assert first == second
