import pytest

from trace_reasoner.datasets.nezha import DEFAULT_ROOT, NezhaDataset, service_of

needs_data = pytest.mark.skipif(
    not (DEFAULT_ROOT / "rca_data").exists(),
    reason="Nezha dataset not cloned to data/Nezha",
)


def test_service_of():
    assert service_of("frontend-579b9bff58-t2dbm") == "frontend"
    assert service_of("cartservice-579f59597d-wc2lz") == "cartservice"
    assert service_of("ts-order-service-7d8fabc-xy12") == "ts-order-service"
    assert service_of("weird") == "weird"


@needs_data
def test_loads_valid_labeled_traces():
    ds = NezhaDataset(system="hipster", max_traces=20, max_traces_per_fault=5, seed=0)
    examples = list(ds)
    assert 0 < len(examples) <= 20
    for ex in examples:
        ids = {s.span_id for s in ex.trace.spans}
        gt = ex.ground_truth
        # every labeled root-cause span exists and belongs to the injected service
        assert gt.root_cause_span_ids
        assert set(gt.root_cause_span_ids) <= ids
        assert all(ex.trace.get(sid).service == gt.root_cause_service for sid in gt.root_cause_span_ids)
        # builds a single-root tree with sane timing
        assert ex.trace.root is not None
        assert ex.trace.duration_ms >= 0


@needs_data
def test_loading_is_deterministic():
    a = [e.trace.trace_id for e in NezhaDataset(system="hipster", max_traces=15, seed=0)]
    b = [e.trace.trace_id for e in NezhaDataset(system="hipster", max_traces=15, seed=0)]
    assert a == b
    assert len(a) == len(set(a))  # no duplicate traces
