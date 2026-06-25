"""The dataset contract every loader implements.

A Dataset yields Examples. An Example pairs a Trace with the GroundTruth of
which span(s) actually caused the anomaly — that label is what the eval harness
scores localizers against.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from trace_reasoner.trace import Trace

# Fault families used across datasets and by the agent's hypothesis labels.
FAULT_FAMILIES = (
    "timeout",
    "contention",
    "saturation",
    "dependency",
    "configuration",
    "exception",
)


@dataclass
class GroundTruth:
    """The known cause of a trace's anomaly.

    `root_cause_span_ids` is a list so it can express a span *chain*, not just a
    single span (Nezha and real incidents often implicate more than one).
    """

    trace_id: str
    root_cause_span_ids: list[str]
    fault_family: str | None = None
    root_cause_service: str | None = None
    notes: str = ""


@dataclass
class Example:
    trace: Trace
    ground_truth: GroundTruth


class Dataset(ABC):
    """Iterable of Examples. Subclasses set `name` and implement iteration."""

    name: str = "dataset"

    @abstractmethod
    def __iter__(self) -> Iterator[Example]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def examples(self) -> list[Example]:
        return list(self)
