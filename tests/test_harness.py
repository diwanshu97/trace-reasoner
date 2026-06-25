from trace_reasoner.baselines.slowest_leaf import slowest_leaf
from trace_reasoner.datasets.synthetic import SyntheticDataset
from trace_reasoner.eval.harness import evaluate


def test_harness_runs_baseline_end_to_end():
    ds = SyntheticDataset(n=40, seed=1)
    rep = evaluate(slowest_leaf, ds, ks=(1, 3), name="slowest_leaf")

    assert rep.n == 40
    assert rep.dataset == "synthetic"
    # top-3 is a superset of top-1, both valid probabilities
    assert 0.0 <= rep.top_k[1] <= rep.top_k[3] <= 1.0
    assert 0.0 <= rep.localization_f1 <= 1.0
    assert 0.0 <= rep.brier <= 1.0


def test_naive_baseline_is_imperfect():
    # The whole point: slowest-leaf cannot solve internal-span / error faults.
    ds = SyntheticDataset(n=60, seed=4)
    rep = evaluate(slowest_leaf, ds, ks=(1, 3))
    assert rep.top_k[3] < 1.0


def test_report_str_is_readable():
    ds = SyntheticDataset(n=5, seed=0)
    text = str(evaluate(slowest_leaf, ds, name="slowest_leaf"))
    assert "slowest_leaf" in text
    assert "top-1 RCA accuracy" in text
