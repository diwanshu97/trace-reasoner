"""Condition B: the multi-agent system (Checkpoint 5 / Module 5).

Three prompt-scoped specialist analysts (latency, dependency, pattern) fan out over one trace
in isolated contexts and write structured Findings into a shared belief state; a deterministic
synthesizer reconciles them into a ranked Prediction and decides on a capped, targeted
re-dispatch. The coordination graph is LangGraph; the synthesizer and the beam-search controller
(Condition C) stay plain Python so the iso-token-budget ablation is unaffected by orchestration.

MultiAgentLocalizer is a Localizer, scored on the same harness as Condition A (ReActLocalizer)
and the slowest_leaf baseline. The LLM is injected: AnthropicClient for live runs,
HeuristicSpecialistClient for offline tests.
"""

from trace_reasoner.multiagent.beam import (
    ToTLocalizer,
    Thought,
    beam_search,
    expand,
    score_thought,
)
from trace_reasoner.multiagent.graph import MultiAgentLocalizer, build_graph
from trace_reasoner.multiagent.mock import HeuristicSpecialistClient
from trace_reasoner.multiagent.specialists import run_specialist
from trace_reasoner.multiagent.state import Finding, MASState
from trace_reasoner.multiagent.synthesizer import reconcile, redispatch_hint, redispatch_targets

__all__ = [
    "MultiAgentLocalizer",
    "build_graph",
    "ToTLocalizer",
    "Thought",
    "beam_search",
    "score_thought",
    "expand",
    "HeuristicSpecialistClient",
    "run_specialist",
    "Finding",
    "MASState",
    "reconcile",
    "redispatch_targets",
    "redispatch_hint",
]
