"""Provider-agnostic LLM interface for the ReAct loop.

The loop never imports a vendor SDK directly — it speaks these dataclasses to an
LLMClient. AnthropicClient implements it against Claude; HeuristicMockClient
implements a deterministic policy for offline tests. This is what lets the same
loop be the research agent and a CI fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Union, runtime_checkable


@dataclass
class ToolSpec:
    """A tool offered to the model (name, description, JSON-schema for inputs)."""

    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AssistantTurn:
    """One model turn: reasoning text plus zero or more tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None


@dataclass
class UserMessage:
    text: str


@dataclass
class ToolResultsMessage:
    results: list[ToolResult]


# A provider-agnostic transcript entry. Each LLMClient converts these to its
# own wire format on every call.
Message = Union[UserMessage, AssistantTurn, ToolResultsMessage]


@runtime_checkable
class LLMClient(Protocol):
    def respond(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> AssistantTurn: ...
