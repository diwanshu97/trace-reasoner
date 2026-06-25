"""Nezha dataset loader (FSE'23, IntelligentDDS/Nezha).

Nezha ships multimodal observability data for two microservice systems —
OnlineBoutique ("hipster", days 2022-08-22/23) and TrainTicket ("ts", days
2023-01-29/30) — under rca_data/<day>/ with log/ metric/ trace/ traceid/
subfolders plus a per-day fault list, e.g. 2022-08-22-fault_list.json.

We use the *trace* modality. Each trace CSV has columns:

    TraceID, SpanID, ParentID, PodName, OperationName,
    StartTimeUnixNano, EndTimeUnixNano, Duration

(ParentID == "root" marks the root; times are epoch nanoseconds; Duration is
microseconds.) PodName is a k8s pod name whose service is the part before the
replicaset/pod hashes (frontend-579b9bff58-t2dbm -> frontend).

Ground truth comes from the fault list. Each injection records inject_time,
inject_pod, and inject_type (cpu_contention / cpu_consumed / network_delay /
exception / return). The file naming aligns faults to traces: a fault injected
at HH:MM maps to the three consecutive minute trace files HH_MM, HH_MM+1,
HH_MM+2 (this triple structure is exactly what the dataset's file list shows).
A trace inside that window has its root cause at the injected service, so we
label every span whose service == injected service as a root-cause span and
drop traces that never touch that service.

Note: the trace modality carries no per-span status/error column (errors live in
the log modality), so span.status is left "OK". Localization here is therefore
latency/structure + label driven, which matches Nezha's own service-level RCA
framing — top-k accuracy ("did we surface a span of the right service?") is the
headline metric for this dataset.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

from trace_reasoner.datasets.base import Dataset, Example, GroundTruth
from trace_reasoner.trace import Span, Trace

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "Nezha"

_SYSTEM_DAYS = {
    "hipster": ["2022-08-22", "2022-08-23"],
    "ts": ["2023-01-29", "2023-01-30"],
}

# Nezha inject_type -> our coarse fault family (raw type kept in GroundTruth.notes).
_FAULT_FAMILY = {
    "cpu_contention": "contention",
    "cpu_consumed": "saturation",
    "network_delay": "dependency",
    "exception": "exception",
    "return": "exception",
}


def service_of(pod_name: str) -> str:
    """k8s pod name -> service. 'frontend-579b9bff58-t2dbm' -> 'frontend'.

    Strips the trailing replicaset-hash and pod-hash segments. Service names may
    themselves contain dashes (e.g. 'ts-order-service'); only the last two
    segments are dropped.
    """
    parts = pod_name.split("-")
    if len(parts) <= 2:
        return pod_name
    return "-".join(parts[:-2])


def _window_files(inject_time: str) -> list[str]:
    """The three consecutive minute trace files for a fault injected at inject_time."""
    dt = datetime.strptime(inject_time, "%Y-%m-%d %H:%M:%S")
    return [(dt + timedelta(minutes=i)).strftime("%H_%M") + "_trace.csv" for i in range(3)]


def _build_trace(trace_id: str, rows: list[dict[str, str]]) -> Trace | None:
    """Build a Trace from CSV rows, or None if the spans don't form one clean tree."""
    t0 = min(int(r["StartTimeUnixNano"]) for r in rows)
    spans: list[Span] = []
    for r in rows:
        start_ns = int(r["StartTimeUnixNano"])
        end_ns = int(r["EndTimeUnixNano"])
        parent = r["ParentID"]
        spans.append(
            Span(
                span_id=r["SpanID"],
                parent_id=None if parent in ("", "root", "None") else parent,
                service=service_of(r["PodName"]),
                operation=r["OperationName"],
                start_ms=(start_ns - t0) / 1e6,
                duration_ms=(end_ns - start_ns) / 1e6,
                status="OK",  # trace modality has no error column
                attributes={"pod": r["PodName"]},
            )
        )
    try:
        return Trace(trace_id=trace_id, spans=spans)
    except ValueError:
        # broken/partial trace (multiple roots, duplicate span ids, ...) — skip it
        return None


def iter_traces_from_csv(path: str | Path) -> Iterator[Trace]:
    """Yield every well-formed Trace in a single Nezha trace CSV (no labels).

    Used by the latency baseline to read fault-free construct_data traces.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            grouped[row["TraceID"]].append(row)
    for trace_id, rows in grouped.items():
        trace = _build_trace(trace_id, rows)
        if trace is not None:
            yield trace


class NezhaDataset(Dataset):
    """Labeled traces from Nezha, normalised to Example(Trace, GroundTruth).

    Sampling is deterministic (seeded). By default it draws a ~200-trace eval set
    spread across faults; tune with max_traces / max_traces_per_fault.
    """

    name = "nezha"

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        system: str = "hipster",
        days: list[str] | None = None,
        max_traces: int = 200,
        max_traces_per_fault: int = 10,
        min_spans: int = 5,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Nezha data not found at {self.root}. Clone it with:\n"
                f"  git clone --depth 1 https://github.com/IntelligentDDS/Nezha.git "
                f"{self.root}"
            )
        if system not in _SYSTEM_DAYS:
            raise ValueError(f"unknown system {system!r}; choose from {list(_SYSTEM_DAYS)}")
        self.system = system
        self.days = days or _SYSTEM_DAYS[system]
        self.max_traces = max_traces
        self.max_traces_per_fault = max_traces_per_fault
        self.min_spans = min_spans
        self.seed = seed
        self._examples: list[Example] | None = None

    def _candidate_ids(self, grouped: dict[str, list[dict]], svc: str) -> list[str]:
        return [
            tid
            for tid, rows in grouped.items()
            if len(rows) >= self.min_spans
            and any(service_of(r["PodName"]) == svc for r in rows)
        ]

    def _load(self) -> list[Example]:
        rng = random.Random(self.seed)
        out: list[Example] = []
        for day in self.days:
            day_dir = self.root / "rca_data" / day
            fault_path = day_dir / f"{day}-fault_list.json"
            if not fault_path.exists():
                continue
            faults_by_hour = json.loads(fault_path.read_text())
            injections = [inj for hour in sorted(faults_by_hour) for inj in faults_by_hour[hour]]

            for inj in injections:
                if len(out) >= self.max_traces:
                    return out
                svc = service_of(inj["inject_pod"])
                family = _FAULT_FAMILY.get(inj["inject_type"])

                # Read window files one at a time; stop once we have enough candidates.
                grouped: dict[str, list[dict]] = defaultdict(list)
                for fname in _window_files(inj["inject_time"]):
                    fpath = day_dir / "trace" / fname
                    if not fpath.exists():
                        continue
                    with fpath.open(newline="") as fh:
                        for row in csv.DictReader(fh):
                            grouped[row["TraceID"]].append(row)
                    if len(self._candidate_ids(grouped, svc)) >= self.max_traces_per_fault:
                        break

                candidate_ids = self._candidate_ids(grouped, svc)
                candidate_ids.sort()  # deterministic before shuffle
                rng.shuffle(candidate_ids)
                candidate_ids = candidate_ids[: self.max_traces_per_fault]

                for tid in candidate_ids:
                    if len(out) >= self.max_traces:
                        return out
                    trace = _build_trace(tid, grouped[tid])
                    if trace is None:
                        continue
                    rc_spans = [s.span_id for s in trace.spans if s.service == svc]
                    if not rc_spans:
                        continue
                    out.append(
                        Example(
                            trace=trace,
                            ground_truth=GroundTruth(
                                trace_id=tid,
                                root_cause_span_ids=rc_spans,
                                fault_family=family,
                                root_cause_service=svc,
                                notes=f"nezha {self.system} {inj['inject_type']} @ {inj['inject_time']}",
                            ),
                        )
                    )
        return out

    def _ensure(self) -> list[Example]:
        if self._examples is None:
            self._examples = self._load()
        return self._examples

    def __len__(self) -> int:
        return len(self._ensure())

    def __iter__(self) -> Iterator[Example]:
        return iter(self._ensure())
