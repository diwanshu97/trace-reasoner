"""The agent: the ReAct loop and its provider-agnostic LLM interface.

The loop (ReActLocalizer) is a Localizer like any baseline — it plugs into
eval.harness.evaluate(). It talks to an LLM through the LLMClient protocol, so
the same loop runs against any of three backends with no code change: a
deterministic mock (HeuristicMockClient), Claude (AnthropicClient), or a local
open model (LocalClient, OpenAI-compatible / Ollama). AnthropicClient and
LocalClient are imported directly from their own modules (kept out of this
__init__ so the package never imports the anthropic or openai SDK unless asked).
"""

from trace_reasoner.agent.llm import (
    AssistantTurn,
    LLMClient,
    ToolCall,
    ToolResult,
    ToolResultsMessage,
    ToolSpec,
    Usage,
    UserMessage,
)
from trace_reasoner.agent.mock import HeuristicMockClient
from trace_reasoner.agent.react import ReActLocalizer, SYSTEM_PROMPT

__all__ = [
    "AssistantTurn",
    "LLMClient",
    "ToolCall",
    "ToolResult",
    "ToolResultsMessage",
    "ToolSpec",
    "Usage",
    "UserMessage",
    "HeuristicMockClient",
    "ReActLocalizer",
    "SYSTEM_PROMPT",
]
