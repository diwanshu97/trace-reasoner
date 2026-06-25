"""baseline_latency: the tool that grounds "is this span actually slow?"

CP2's argument: a frontier model can guess that a 1100ms Redis span is slow, but
"plausible" isn't "right" — some ops are normally that slow. The only fix is to
look up the real per-(service, operation) distribution. This module builds that
distribution from fault-free traces (synthetic normals, or Nezha construct_data)
and returns a grounded verdict.

A span is judged anomalous when its observed exclusive time exceeds the p95 of
its (service, operation) baseline *and* sits >=2 sigma above the mean — both,
so a naturally high-variance op doesn't trip on a single slow sample.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev

from trace_reasoner.trace import Trace


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass
class LatencyStats:
    count: int
    p50: float
    p95: float
    mean: float
    std: float


@dataclass
class Verdict:
    service: str
    operation: str
    observed_ms: float
    known: bool          # was this (service, operation) seen in the baseline?
    is_anomalous: bool
    p50: float = 0.0
    p95: float = 0.0
    z: float = 0.0

    def __str__(self) -> str:
        if not self.known:
            return f"{self.service}/{self.operation}: no baseline (observed {self.observed_ms:.1f}ms)"
        tag = "ANOMALOUS" if self.is_anomalous else "normal"
        return (
            f"{self.service}/{self.operation}: {tag} — observed {self.observed_ms:.1f}ms "
            f"(p50 {self.p50:.1f}, p95 {self.p95:.1f}, z={self.z:.1f})"
        )


class LatencyBaseline:
    """Per-(service, operation) exclusive-time distribution from fault-free traces."""

    def __init__(self, stats: dict[tuple[str, str], LatencyStats]) -> None:
        self._stats = stats

    def __len__(self) -> int:
        return len(self._stats)

    def keys(self) -> list[tuple[str, str]]:
        return list(self._stats)

    @classmethod
    def from_traces(cls, traces: Iterable[Trace], use_self_time: bool = True) -> "LatencyBaseline":
        samples: dict[tuple[str, str], list[float]] = defaultdict(list)
        for t in traces:
            for s in t.spans:
                value = t.self_time_ms(s.span_id) if use_self_time else s.duration_ms
                samples[(s.service, s.operation)].append(value)
        stats: dict[tuple[str, str], LatencyStats] = {}
        for key, vals in samples.items():
            vals.sort()
            stats[key] = LatencyStats(
                count=len(vals),
                p50=_percentile(vals, 50),
                p95=_percentile(vals, 95),
                mean=fmean(vals),
                std=pstdev(vals) if len(vals) > 1 else 0.0,
            )
        return cls(stats)

    @classmethod
    def from_nezha(
        cls,
        root: str | Path | None = None,
        system: str = "hipster",
        max_files: int = 6,
        use_self_time: bool = True,
    ) -> "LatencyBaseline":
        """Build a baseline from Nezha's fault-free construct_data traces."""
        from trace_reasoner.datasets.nezha import (
            DEFAULT_ROOT,
            _SYSTEM_DAYS,
            iter_traces_from_csv,
        )

        base = Path(root) if root is not None else DEFAULT_ROOT
        traces: list[Trace] = []
        read = 0
        for day in _SYSTEM_DAYS[system]:
            tdir = base / "construct_data" / day / "trace"
            if not tdir.exists():
                continue
            for fpath in sorted(tdir.glob("*_trace.csv")):
                if read >= max_files:
                    break
                traces.extend(iter_traces_from_csv(fpath))
                read += 1
        return cls.from_traces(traces, use_self_time=use_self_time)

    def verdict(self, service: str, operation: str, observed_ms: float) -> Verdict:
        st = self._stats.get((service, operation))
        if st is None:
            return Verdict(service, operation, round(observed_ms, 3), known=False, is_anomalous=False)
        z = (observed_ms - st.mean) / st.std if st.std > 0 else 0.0
        if st.std > 0:
            is_anom = observed_ms > st.p95 and z >= 2.0
        else:
            is_anom = observed_ms > st.p95
        return Verdict(
            service=service,
            operation=operation,
            observed_ms=round(observed_ms, 3),
            known=True,
            is_anomalous=is_anom,
            p50=round(st.p50, 3),
            p95=round(st.p95, 3),
            z=round(z, 2),
        )


def baseline_latency(
    baseline: LatencyBaseline, trace: Trace, span_id: str, use_self_time: bool = True
) -> Verdict:
    """Tool wrapper: judge one span of a trace against the baseline."""
    s = trace.get(span_id)
    observed = trace.self_time_ms(span_id) if use_self_time else s.duration_ms
    return baseline.verdict(s.service, s.operation, observed)
