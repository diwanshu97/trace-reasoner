"""Evaluation: localizer prediction types, metrics, and the scoring harness."""

from trace_reasoner.eval.metrics import Hypothesis, Prediction
from trace_reasoner.eval.harness import EvalReport, evaluate

__all__ = ["Hypothesis", "Prediction", "EvalReport", "evaluate"]
