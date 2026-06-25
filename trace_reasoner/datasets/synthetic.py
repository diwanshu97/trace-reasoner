"""Synthetic trace generator with a known injected anomaly.

Why this exists before the real data: it gives ground-truth labels for free, is
fully deterministic (seeded), needs no download, and is the fixture the eval
harness and tools are tested against. It is also the substrate for trajectory
mining later (Week 4).

Timing model: each span does its own `self_time` of work, then runs its children
sequentially. So a span's duration = self_time + sum(child durations), and the
whole subtree lies on the critical path. A latency fault inflates one span's
self_time; an error fault marks a span ERROR and propagates the error up its
ancestry (as a failed call would).

Note the deliberate trap for naive localizers: the injected span is often an
*internal* node with high self-time, while the slowest *leaf* (by total duration)
is somewhere else. "Blame the slowest leaf" therefore gets these wrong — which
is exactly the failure mode the agent is meant to beat.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from trace_reasoner.datasets.base import Dataset, Example, GroundTruth
from trace_reasoner.trace import Span, Trace

_SERVICES = [
    "gateway", "cart", "checkout", "payment", "catalog",
    "inventory", "redis", "db", "auth", "shipping",
]
_OPS = ["GET", "POST", "query", "rpc", "commit", "lookup", "acquire", "publish"]

_LATENCY_FAMILIES = ["saturation", "contention"]
_ERROR_FAMILIES = ["exception", "dependency"]


@dataclass
class _Node:
    span_id: str
    parent_id: str | None
    service: str
    operation: str
    self_time: float
    status: str = "OK"
    children: list["_Node"] = field(default_factory=list)


def generate_example(
    seed: int,
    depth: int = 4,
    branching: int = 3,
    fault: str = "latency",
    trace_id: str | None = None,
) -> Example:
    """Generate one (Trace, GroundTruth) pair with a single injected fault."""
    rng = random.Random(seed)
    tid = trace_id or f"syn-{seed:06d}"
    counter = {"n": 0}

    def new_id() -> str:
        counter["n"] += 1
        return f"{tid}-s{counter['n']}"

    def build(parent_id: str | None, level: int) -> _Node:
        node = _Node(
            span_id=new_id(),
            parent_id=parent_id,
            service=rng.choice(_SERVICES),
            operation=rng.choice(_OPS),
            self_time=rng.uniform(5.0, 40.0),
        )
        if level < depth:
            for _ in range(rng.randint(1, branching)):
                node.children.append(build(node.span_id, level + 1))
        return node

    root = build(None, 0)

    all_nodes: list[_Node] = []

    def collect(n: _Node) -> None:
        all_nodes.append(n)
        for c in n.children:
            collect(c)

    collect(root)
    by_id = {n.span_id: n for n in all_nodes}

    target: _Node | None = None
    family: str | None = None
    if fault in ("none", None):
        pass  # fault-free trace (used for latency baselines / negatives)
    elif fault == "latency":
        target = rng.choice(all_nodes[1:])  # never the root
        target.self_time *= rng.uniform(6.0, 12.0)
        family = rng.choice(_LATENCY_FAMILIES)
    elif fault == "error":
        target = rng.choice(all_nodes[1:])
        target.status = "ERROR"
        cur = target
        while cur.parent_id is not None:
            cur = by_id[cur.parent_id]
            cur.status = "ERROR"
        family = rng.choice(_ERROR_FAMILIES)
    else:
        raise ValueError(f"unknown fault mode: {fault!r}")

    # Durations bottom-up (memoised), start times top-down.
    dur_cache: dict[str, float] = {}

    def duration(n: _Node) -> float:
        if n.span_id not in dur_cache:
            dur_cache[n.span_id] = n.self_time + sum(duration(c) for c in n.children)
        return dur_cache[n.span_id]

    spans: list[Span] = []

    def emit(n: _Node, start: float) -> None:
        spans.append(
            Span(
                span_id=n.span_id,
                parent_id=n.parent_id,
                service=n.service,
                operation=n.operation,
                start_ms=round(start, 3),
                duration_ms=round(duration(n), 3),
                status=n.status,
            )
        )
        t = start + n.self_time  # self work first, then children sequentially
        for c in n.children:
            emit(c, t)
            t += duration(c)

    emit(root, 0.0)

    trace = Trace(trace_id=tid, spans=spans)
    gt = GroundTruth(
        trace_id=tid,
        root_cause_span_ids=[target.span_id] if target else [],
        fault_family=family,
        root_cause_service=target.service if target else None,
        notes=f"injected {fault} fault" if target else "no fault",
    )
    return Example(trace=trace, ground_truth=gt)


class SyntheticDataset(Dataset):
    """A deterministic stream of synthetic Examples mixing latency and error faults."""

    name = "synthetic"

    def __init__(
        self,
        n: int = 50,
        seed: int = 0,
        error_ratio: float = 0.3,
        depth: int = 4,
        branching: int = 3,
    ) -> None:
        self.n = n
        self.seed = seed
        self.error_ratio = error_ratio
        self.depth = depth
        self.branching = branching

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[Example]:
        picker = random.Random(self.seed)
        for i in range(self.n):
            fault = "error" if picker.random() < self.error_ratio else "latency"
            yield generate_example(
                seed=self.seed * 100_000 + i,
                depth=self.depth,
                branching=self.branching,
                fault=fault,
            )


def normal_traces(
    n: int, seed: int = 0, depth: int = 4, branching: int = 3
) -> list[Trace]:
    """Fault-free traces, for building latency baselines or as negatives."""
    return [
        generate_example(
            seed=seed * 100_000 + i, depth=depth, branching=branching, fault="none"
        ).trace
        for i in range(n)
    ]
