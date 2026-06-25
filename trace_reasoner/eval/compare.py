"""Build the A/B/C conditions and run them side-by-side on one ruler.

This is the engine behind the central experiment: three conditions at an equal token budget —
A monolithic ReAct, B multi-agent specialists, C = B + Tree-of-Thought beam search — each scored
on the same harness over the same dataset. `build_conditions` constructs the three localizers
(optionally each wrapped in the Checkpoint 6 SafeLocalizer); `compare` evaluates them and returns
one report per condition. The offline mocks are the default LLM, so the whole comparison runs with
no API key — swap in AnthropicClient for the live run.

Both the CLI (eval_conditions.py) and the demo UI call this, so they always agree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from trace_reasoner.agent.mock import HeuristicMockClient
from trace_reasoner.agent.react import ReActLocalizer
from trace_reasoner.datasets.base import Dataset
from trace_reasoner.datasets.synthetic import SyntheticDataset, normal_traces
from trace_reasoner.eval.harness import EvalReport, evaluate
from trace_reasoner.multiagent.beam import ToTLocalizer
from trace_reasoner.multiagent.graph import MultiAgentLocalizer
from trace_reasoner.multiagent.mock import HeuristicSpecialistClient
from trace_reasoner.safety.router import RouterThresholds, SafeLocalizer
from trace_reasoner.tools.baseline import LatencyBaseline

CONDITION_LABELS = {
    "A": "A — monolithic ReAct",
    "B": "B — multi-agent specialists",
    "C": "C — multi-agent + Tree-of-Thought",
}


@dataclass
class Conditions:
    """The three task-condition localizers, keyed A/B/C, plus the baseline they share."""

    localizers: dict[str, Callable]
    baseline: LatencyBaseline


def build_conditions(
    baseline: LatencyBaseline | None = None,
    baseline_n: int = 100,
    baseline_seed: int = 0,
    live_llm=None,
    safe: bool = False,
    thresholds: RouterThresholds | None = None,
) -> Conditions:
    """Construct Conditions A, B, and C over a shared latency baseline.

    `live_llm` injects a real LLMClient (AnthropicClient) into all three; left None, each condition
    runs on its deterministic offline mock so the comparison needs no API key. `safe=True` wraps
    every condition in the Checkpoint 6 SafeLocalizer (guardrails + verification + trust/risk
    router), so the same comparison can be run with the control system on or off.
    """
    if baseline is None:
        baseline = LatencyBaseline.from_traces(normal_traces(baseline_n, seed=baseline_seed))

    # The mocks are role-specific: ReAct uses HeuristicMockClient; the specialists (B and C's
    # generators) use HeuristicSpecialistClient. A live LLM replaces both.
    react_llm = live_llm or HeuristicMockClient()
    specialist_llm = live_llm or HeuristicSpecialistClient()

    localizers: dict[str, Callable] = {
        "A": ReActLocalizer(react_llm, baseline),
        "B": MultiAgentLocalizer(specialist_llm, baseline),
        "C": ToTLocalizer(specialist_llm, baseline),
    }

    if safe:
        localizers = {
            key: SafeLocalizer(inner, baseline, thresholds or RouterThresholds())
            for key, inner in localizers.items()
        }

    return Conditions(localizers=localizers, baseline=baseline)


def compare(
    dataset: Dataset,
    conditions: Conditions | None = None,
    ks: tuple[int, ...] = (1, 3),
) -> dict[str, EvalReport]:
    """Evaluate every condition on the dataset; return {condition_key: EvalReport}."""
    conditions = conditions or build_conditions()
    return {
        key: evaluate(localizer, dataset, ks=ks, name=CONDITION_LABELS.get(key, key))
        for key, localizer in conditions.localizers.items()
    }


def comparison_table(reports: dict[str, EvalReport]) -> str:
    """Render the per-condition reports as one aligned comparison table (the headline artifact)."""
    header = f"{'cond':<5} {'top-1':>6} {'top-3':>6} {'F1':>6} {'Brier':>6} {'ECE':>6} {'escal':>6}"
    lines = [header, "-" * len(header)]
    for key in sorted(reports):
        r = reports[key]
        lines.append(
            f"{key:<5} "
            f"{r.top_k.get(1, 0.0):>6.3f} {r.top_k.get(3, 0.0):>6.3f} "
            f"{r.localization_f1:>6.3f} {r.brier:>6.3f} {r.ece:>6.3f} {r.escalation:>6.3f}"
        )
    return "\n".join(lines)
