import pytest

from trace_reasoner.datasets.base import GroundTruth
from trace_reasoner.eval.metrics import (
    Hypothesis,
    Prediction,
    brier_score,
    escalation_rate,
    expected_calibration_error,
    localization_f1,
    selective_accuracy_coverage,
    top_k_accuracy,
)


def pred(trace_id, ids_conf):
    return Prediction(trace_id, [Hypothesis(i, c) for i, c in ids_conf])


def gt(trace_id, ids):
    return GroundTruth(trace_id, list(ids))


def test_top_k_accuracy():
    pairs = [
        (pred("t1", [("a", 0.9), ("b", 0.1)]), gt("t1", ["b"])),  # b is rank 2
        (pred("t2", [("x", 0.8), ("y", 0.2)]), gt("t2", ["z"])),  # never predicted
    ]
    assert top_k_accuracy(pairs, 1) == 0.0
    assert top_k_accuracy(pairs, 2) == 0.5


def test_localization_f1_hit_and_miss():
    assert localization_f1([(pred("t", [("a", 0.9)]), gt("t", ["a"]))]) == 1.0
    assert localization_f1([(pred("t", [("a", 0.9)]), gt("t", ["b"]))]) == 0.0


def test_localization_f1_partial_overlap():
    # true chain {a, b}; predict top-2 {a, x} -> tp=1, prec .5, rec .5, f1 .5
    p = pred("t", [("a", 0.9), ("x", 0.5), ("b", 0.1)])
    assert localization_f1([(p, gt("t", ["a", "b"]))]) == pytest.approx(0.5)


def test_brier_score():
    pairs = [
        (pred("t1", [("a", 1.0)]), gt("t1", ["a"])),  # correct, conf 1 -> 0
        (pred("t2", [("x", 1.0)]), gt("t2", ["y"])),  # wrong,  conf 1 -> 1
    ]
    assert brier_score(pairs) == pytest.approx(0.5)


def test_empty_inputs_are_safe():
    assert top_k_accuracy([], 1) == 0.0
    assert localization_f1([]) == 0.0
    assert brier_score([]) == 0.0
    assert expected_calibration_error([]) == 0.0
    assert escalation_rate([]) == 0.0


# --- Checkpoint 6: calibration / selective-prediction metrics -----------------
def test_ece_zero_when_perfectly_calibrated():
    # Two confident-correct and two unconfident-incorrect: in each bin conf == acc -> ECE 0.
    pairs = [
        (pred("t1", [("a", 1.0)]), gt("t1", ["a"])),  # conf 1.0, correct
        (pred("t2", [("b", 1.0)]), gt("t2", ["b"])),  # conf 1.0, correct
        (pred("t3", [("c", 0.0)]), gt("t3", ["z"])),  # conf 0.0, wrong
        (pred("t4", [("d", 0.0)]), gt("t4", ["z"])),  # conf 0.0, wrong
    ]
    assert expected_calibration_error(pairs, bins=10) == pytest.approx(0.0)


def test_ece_penalizes_overconfidence():
    # Confidence 1.0 but only half correct -> ECE ~ 0.5 (one bin, |1.0 - 0.5|).
    pairs = [
        (pred("t1", [("a", 1.0)]), gt("t1", ["a"])),  # correct
        (pred("t2", [("b", 1.0)]), gt("t2", ["z"])),  # wrong
    ]
    assert expected_calibration_error(pairs, bins=10) == pytest.approx(0.5)


def test_ece_ignores_abstentions():
    # An abstention carries no confidence; ECE is computed only over answered traces.
    pairs = [
        (pred("t1", [("a", 1.0)]), gt("t1", ["a"])),  # answered, calibrated
        (Prediction("t2", []), gt("t2", ["z"])),       # abstained
    ]
    assert expected_calibration_error(pairs, bins=10) == pytest.approx(0.0)


def test_escalation_rate_counts_abstentions():
    pairs = [
        (pred("t1", [("a", 0.9)]), gt("t1", ["a"])),
        (Prediction("t2", []), gt("t2", ["z"])),
        (Prediction("t3", []), gt("t3", ["y"])),
    ]
    assert escalation_rate(pairs) == pytest.approx(2 / 3)


def test_selective_accuracy_rises_as_coverage_falls():
    # One wrong-but-low-confidence answer and one right-high-confidence answer:
    # raising the threshold drops the wrong one, so accuracy goes up as coverage drops.
    pairs = [
        (pred("t1", [("a", 0.9)]), gt("t1", ["a"])),  # right, confident
        (pred("t2", [("b", 0.2)]), gt("t2", ["z"])),  # wrong, unconfident
    ]
    curve = selective_accuracy_coverage(pairs, thresholds=(0.0, 0.5))
    cov0, acc0 = curve[0][1], curve[0][2]
    cov1, acc1 = curve[1][1], curve[1][2]
    assert cov0 == 1.0 and acc0 == pytest.approx(0.5)  # answer both -> 50%
    assert cov1 == 0.5 and acc1 == pytest.approx(1.0)  # keep only the confident one -> 100%
