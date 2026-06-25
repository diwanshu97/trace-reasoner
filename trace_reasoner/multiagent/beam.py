"""Condition C: Tree-of-Thought beam search over root-cause hypotheses (Checkpoint 4).

Condition C is Condition B with a beam search spliced into the hypothesize phase. The
specialist analysts are the *thought generators* (Checkpoint 4's mapping): each proposes a
candidate culprit span. A separate *critic* scores every hypothesis against the four criteria
from Checkpoint 4 — anomaly grounding, critical-path coverage, precedent support, verification —
using the same grounded tools the specialists use, never raw duration. The *controller* is plain
Python (the "decision maker as deterministic Python, not a model" from Checkpoint 4), so the beam
adds zero hidden LLM calls and the iso-token-budget ablation against A and B stays honest: total
tool calls is the controlled variable, and beam width x depth is the knob that sets it.

Generation and evaluation live in separate hands — that separation is what makes this
Tree-of-Thought rather than self-consistency over one chain. The beam keeps the top `beam_width`
hypotheses alive at each depth, develops survivors by walking one structural hop (the
Redis-vs-pool-saturation reframe), re-scores, and prunes back. A branch is hard-pruned only by a
negative tool result (baseline says the span is normal), never by a weak early score, so the
subtle true cause is never killed before retrieval can reframe it. If nothing clears the
confidence floor, the answer is "inconclusive within budget" — the honest failure mode from
Checkpoint 1.

`ToTLocalizer` is a Localizer — `tot(trace) -> Prediction` — so it scores on the same harness as
Conditions A and B. It calls the specialists directly (no LangGraph), so the offline
HeuristicSpecialistClient drives the whole search in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trace_reasoner.agent.llm import LLMClient
from trace_reasoner.eval.metrics import Hypothesis, Prediction
from trace_reasoner.multiagent.specialists import run_specialist
from trace_reasoner.rag.retriever import PrecedentRetriever
from trace_reasoner.tools.baseline import LatencyBaseline, baseline_latency
from trace_reasoner.tools.walk_tree import survey
from trace_reasoner.trace import Trace

# Critic criteria weights (Checkpoint 4). Verified evidence (a confirmed baseline anomaly) and
# anomaly grounding carry the most weight; coverage and precedent are supporting signals. This is
# the verified-over-advisory rule from Checkpoint 3, made into a scoring vector.
_W_ANOMALY = 0.35
_W_COVERAGE = 0.25
_W_PRECEDENT = 0.20
_W_VERIFIED = 0.20

_STOP_SCORE = 0.6       # a verified leader at/above this can stop the search early
_STOP_MARGIN = 0.15     # ...if it also leads the runner-up by this much
_PRUNE_FLOOR = 0.30     # below this and outside the beam, a branch is dropped (also the answer floor)


@dataclass
class Thought:
    """One candidate root-cause hypothesis: a span, a fault family, and its accumulated score.

    `verified` is set when baseline_latency confirms the span is genuinely anomalous (the
    verified evidence of Checkpoint 3); `hard_pruned` when baseline reports it normal — a negative
    tool result that kills the branch regardless of how confidently a lens named it.
    """

    span_id: str
    fault_family: str | None = None
    depth: int = 1
    score: float = 0.0
    verified: bool = False
    anomalous: bool | None = None
    hard_pruned: bool = False
    evidence: list[str] = field(default_factory=list)
    criteria: dict[str, float] = field(default_factory=dict)


@dataclass
class _Budget:
    """Bounds the critic's tool calls, so the beam stays inside one trace's compute envelope."""

    limit: int
    calls: int = 0

    def spend(self, n: int = 1) -> None:
        self.calls += n

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.limit


def _seed_from_survey(trace: Trace, top: int) -> list[Thought]:
    """The post-survey root state: the hottest-by-self-time spans as initial hypotheses.

    Seeding from self-time (not total duration) is the deliberate counter to the "blame the
    slowest leaf" trap — the injected fault is usually an internal node with high *exclusive*
    time, which survey surfaces directly.
    """
    s = survey(trace, top=top)
    return [
        Thought(span_id=v.span_id, depth=1, evidence=[f"survey: hottest self-time {v.self_time_ms}ms"])
        for v in s.hottest_by_self_time
    ]


def score_thought(
    thought: Thought,
    trace: Trace,
    baseline: LatencyBaseline,
    retriever: PrecedentRetriever | None,
    budget: _Budget | None = None,
) -> Thought:
    """The critic: fold the four criteria into a value in [0, 1] from grounded tool results.

    Anomaly grounding and verification come from baseline_latency; coverage from the span's share
    of trace wall-time and whether it sits on the critical path; precedent from the retriever. A
    span baseline reports normal is hard-pruned (a negative tool result), never merely down-scored.
    """
    span = trace.get(thought.span_id)
    verdict = baseline_latency(baseline, trace, thought.span_id)
    if budget is not None:
        budget.spend()

    if verdict.known and not verdict.is_anomalous:
        # Negative tool result — hard prune (Checkpoint 4). Not anomalous means not the cause.
        thought.hard_pruned = True
        thought.anomalous = False
        thought.score = 0.0
        thought.criteria = {"anomaly": 0.0, "coverage": 0.0, "precedent": 0.0, "verified": 0.0}
        thought.evidence.append(f"baseline: {verdict}")
        return thought

    if verdict.known and verdict.is_anomalous:
        anomaly = min(1.0, max(0.6, verdict.z / 4.0))  # confirmed anomaly; z scales the strength
        verified = True
    else:
        anomaly = 0.3  # unknown (service, op): can't ground, so weak advisory credit only
        verified = False

    coverage = min(1.0, trace.self_time_ms(thought.span_id) / trace.duration_ms) if trace.duration_ms else 0.0
    if thought.span_id in {sp.span_id for sp in trace.critical_path()}:
        coverage = min(1.0, coverage + 0.1)  # on the latency-gating path

    precedent = 0.0
    if retriever is not None:
        query = f"{span.service} {span.operation} {thought.fault_family or ''} high self-time latency anomaly"
        precs = retriever.retrieve(query, k=5)
        if budget is not None:
            budget.spend()
        if precs:
            top = precs[0]
            match = 1.0 if (thought.fault_family and top.fault_family == thought.fault_family) else 0.6
            precedent = min(1.0, top.score * match)
            thought.evidence.append(f"precedent {top.id} ({top.fault_family}) cos={top.score}")

    value = _W_ANOMALY * anomaly + _W_COVERAGE * coverage + _W_PRECEDENT * precedent + _W_VERIFIED * (1.0 if verified else 0.0)
    thought.anomalous = verdict.is_anomalous if verdict.known else None
    thought.verified = verified
    thought.score = round(min(1.0, value), 3)
    thought.criteria = {
        "anomaly": round(anomaly, 3),
        "coverage": round(coverage, 3),
        "precedent": round(precedent, 3),
        "verified": 1.0 if verified else 0.0,
    }
    thought.evidence.append(f"baseline: {verdict}")
    return thought


def expand(thought: Thought, trace: Trace, branching: int) -> list[Thought]:
    """Develop a surviving hypothesis by walking one structural hop (Checkpoint 4 branching).

    Refine *up* to the parent (the "the slow leaf is only slow because the pool one hop up is
    saturated" reframe) and *down* into the hottest-self-time child (a span blocked on a child).
    Both are grounded structural moves via the span tree, capped at `branching`.
    """
    children: list[Thought] = []
    parent = trace.parent(thought.span_id)
    if parent is not None:
        children.append(
            Thought(span_id=parent.span_id, fault_family="saturation", depth=thought.depth + 1,
                    evidence=[f"refine up: parent of {thought.span_id}"])
        )
    kids = trace.children(thought.span_id)
    if kids:
        hottest = max(kids, key=lambda c: trace.self_time_ms(c.span_id))
        children.append(
            Thought(span_id=hottest.span_id, fault_family=thought.fault_family, depth=thought.depth + 1,
                    evidence=[f"refine down: hottest child of {thought.span_id}"])
        )
    return children[:branching]


def _dedup(thoughts: list[Thought]) -> list[Thought]:
    """Collapse to one thought per span, keeping the highest-scored (Checkpoint 4 duplicate prune)."""
    best: dict[str, Thought] = {}
    for t in thoughts:
        cur = best.get(t.span_id)
        if cur is None or t.score > cur.score:
            best[t.span_id] = t
    return list(best.values())


def _can_stop(survivors: list[Thought]) -> bool:
    """Stop early when a verified leader clears the confidence floor and leads by a margin."""
    if not survivors:
        return False
    lead = survivors[0]
    if not (lead.verified and lead.score >= _STOP_SCORE):
        return False
    return len(survivors) < 2 or (lead.score - survivors[1].score) >= _STOP_MARGIN


def beam_search(
    trace: Trace,
    baseline: LatencyBaseline,
    retriever: PrecedentRetriever | None,
    generator,
    beam_width: int = 3,
    max_depth: int = 4,
    branching: int = 4,
    max_tool_calls: int = 24,
    confidence_floor: float = _PRUNE_FLOOR,
) -> Prediction:
    """Run the beam: generate, score, prune, expand survivors, until verified or budget-bound.

    `generator(trace) -> list[Thought]` supplies the depth-1 frontier (the LLM specialists). The
    survey suspects are added so the search has breadth even if a generator misses. Returns a
    ranked top-3 Prediction, or an empty one ("inconclusive within budget") when nothing clears
    the floor.
    """
    budget = _Budget(limit=max_tool_calls)
    frontier = _dedup(generator(trace) + _seed_from_survey(trace, top=branching))
    survivors: list[Thought] = []

    for depth in range(1, max_depth + 1):
        scored: list[Thought] = []
        for t in frontier:
            t.depth = depth
            score_thought(t, trace, baseline, retriever, budget)
            if not t.hard_pruned:  # baseline-normal branches are dropped outright
                scored.append(t)
            if budget.exhausted:
                break

        pool = _dedup(scored + survivors)
        pool.sort(key=lambda x: x.score, reverse=True)
        survivors = pool[:beam_width]

        if _can_stop(survivors) or budget.exhausted or depth == max_depth:
            break

        # Expand survivors by one structural hop; only pursue spans not already in the beam.
        seen = {t.span_id for t in survivors}
        frontier = [t for t in _dedup([c for t in survivors for c in expand(t, trace, branching)]) if t.span_id not in seen]
        if not frontier:
            break

    ranked = [
        Hypothesis(span_id=t.span_id, confidence=t.score, fault_family=t.fault_family, evidence=t.evidence)
        for t in survivors
        if t.score >= confidence_floor
    ]
    ranked.sort(key=lambda h: h.confidence, reverse=True)
    return Prediction(trace_id=trace.trace_id, ranked=ranked[:3])


def _specialist_generator(llm: LLMClient, baseline: LatencyBaseline, retriever, max_steps: int):
    """Default thought generator: the three Condition B specialists, each proposing one hypothesis."""

    def generate(trace: Trace) -> list[Thought]:
        thoughts: list[Thought] = []
        for role in ("latency", "dependency", "pattern"):
            f = run_specialist(role, llm, trace, baseline, retriever=retriever, max_steps=max_steps)
            if f.span_id:
                rationale = f.evidence[0] if f.evidence else "proposed"
                thoughts.append(
                    Thought(span_id=f.span_id, fault_family=f.fault_family, depth=1,
                            evidence=[f"{role} analyst: {rationale}"])
                )
        return thoughts

    return generate


class ToTLocalizer:
    """Condition C: the Condition B specialists as generators + a Tree-of-Thought beam controller.

    A Localizer like ReActLocalizer (A) and MultiAgentLocalizer (B) — scored on the same harness,
    so the A/B/C iso-budget ablation is measured on one ruler. The LLM, baseline, and optional
    retriever are injected exactly as for Condition B. `max_tool_calls` is the beam's deterministic
    tool budget; beam_width x max_depth is how the controlled variable (total tool calls) is set.
    """

    def __init__(
        self,
        llm: LLMClient,
        baseline: LatencyBaseline,
        retriever: PrecedentRetriever | None = None,
        beam_width: int = 3,
        max_depth: int = 4,
        branching: int = 4,
        specialist_max_steps: int = 8,
        max_tool_calls: int = 24,
        confidence_floor: float = _PRUNE_FLOOR,
    ) -> None:
        self._llm = llm
        self._baseline = baseline
        self._retriever = retriever
        self._beam_width = beam_width
        self._max_depth = max_depth
        self._branching = branching
        self._specialist_max_steps = specialist_max_steps
        self._max_tool_calls = max_tool_calls
        self._confidence_floor = confidence_floor

    def __call__(self, trace: Trace) -> Prediction:
        generator = _specialist_generator(self._llm, self._baseline, self._retriever, self._specialist_max_steps)
        return beam_search(
            trace,
            self._baseline,
            self._retriever,
            generator,
            beam_width=self._beam_width,
            max_depth=self._max_depth,
            branching=self._branching,
            max_tool_calls=self._max_tool_calls,
            confidence_floor=self._confidence_floor,
        )
