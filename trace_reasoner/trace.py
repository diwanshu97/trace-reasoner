"""Canonical trace representation for Trace-Reasoner.

Every tool the agent uses (walk_tree, baseline_latency, ...) operates on the
Span/Trace types defined here. Keeping one model means the dataset loaders
(synthetic, Nezha, DeathStarBench, Alibaba) all normalise to the same shape and
the agent code never has to know a vendor-specific schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Span:
    """One unit of work in a trace. Times are in milliseconds.

    `start_ms` is relative to the trace start (so the root begins at 0).
    """

    span_id: str
    parent_id: str | None
    service: str
    operation: str
    start_ms: float
    duration_ms: float
    status: str = "OK"  # "OK" or "ERROR"
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms

    @property
    def is_error(self) -> bool:
        return self.status.upper() == "ERROR"


@dataclass
class Trace:
    """A single distributed trace: a set of spans forming one tree.

    Builds parent/child indices on construction and validates that the spans
    form exactly one rooted tree.
    """

    trace_id: str
    spans: list[Span]

    _by_id: dict[str, Span] = field(init=False, repr=False)
    _children: dict[str, list[str]] = field(init=False, repr=False)
    _root_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {s.span_id: s for s in self.spans}
        if len(self._by_id) != len(self.spans):
            raise ValueError(f"duplicate span_id in trace {self.trace_id}")

        self._children = {s.span_id: [] for s in self.spans}
        roots: list[str] = []
        for s in self.spans:
            if s.parent_id is None or s.parent_id not in self._by_id:
                roots.append(s.span_id)
            else:
                self._children[s.parent_id].append(s.span_id)

        if len(roots) != 1:
            raise ValueError(
                f"trace {self.trace_id} must have exactly one root, found {len(roots)}"
            )
        self._root_id = roots[0]

    # --- lookups -----------------------------------------------------------
    def get(self, span_id: str) -> Span:
        return self._by_id[span_id]

    @property
    def root(self) -> Span:
        return self._by_id[self._root_id]

    def children(self, span_id: str) -> list[Span]:
        return [self._by_id[c] for c in self._children[span_id]]

    def parent(self, span_id: str) -> Span | None:
        pid = self._by_id[span_id].parent_id
        return self._by_id[pid] if pid in self._by_id else None

    def ancestors(self, span_id: str) -> list[Span]:
        """Parent, grandparent, ... up to (and including) the root."""
        out: list[Span] = []
        cur = self.parent(span_id)
        while cur is not None:
            out.append(cur)
            cur = self.parent(cur.span_id)
        return out

    def descendants(self, span_id: str) -> list[Span]:
        out: list[Span] = []
        stack = list(self._children[span_id])
        while stack:
            cid = stack.pop()
            out.append(self._by_id[cid])
            stack.extend(self._children[cid])
        return out

    def is_leaf(self, span_id: str) -> bool:
        return not self._children[span_id]

    def leaves(self) -> list[Span]:
        return [s for s in self.spans if self.is_leaf(s.span_id)]

    # --- latency structure -------------------------------------------------
    def self_time_ms(self, span_id: str) -> float:
        """Exclusive time: span duration minus time attributable to children.

        Uses the union of child intervals, so overlapping (parallel) children
        are not double-counted. Clipped at 0.
        """
        span = self._by_id[span_id]
        kids = self.children(span_id)
        if not kids:
            return span.duration_ms
        covered = _union_length([(c.start_ms, c.end_ms) for c in kids])
        return max(0.0, span.duration_ms - covered)

    @property
    def duration_ms(self) -> float:
        """End-to-end latency of the trace."""
        return self.root.duration_ms

    def critical_path(self) -> list[Span]:
        """Root-to-leaf path following the latest-finishing child at each step.

        A heuristic approximation of the path that gates end-to-end latency:
        at each node, descend into the child that finishes last.
        """
        path = [self.root]
        cur = self.root
        while True:
            kids = self.children(cur.span_id)
            if not kids:
                break
            cur = max(kids, key=lambda s: s.end_ms)
            path.append(cur)
        return path


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by a set of [start, end] intervals (merging overlaps)."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    total += cur_end - cur_start
    return total
