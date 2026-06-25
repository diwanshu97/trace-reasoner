"""LLMClient backed by a local open-weight model via an OpenAI-compatible endpoint.

The third brain behind the agent (alongside AnthropicClient and HeuristicMockClient): a small
open model — Qwen2.5-7B, Llama-3.1-8B — served locally by Ollama, llama.cpp, or LM Studio, all of
which speak the OpenAI chat-completions wire format. Targeting that format (not raw transformers)
means one client works across every local server and reuses the tool-calling protocol small models
are least bad at.

Why it exists for the capstone: it adds a *model-capability axis* to the A/B/C ablation — do the
architecture gains (C > B > A on calibration) survive when the underlying model is an order of
magnitude smaller than Claude? It implements the same `LLMClient.respond`, so it drops into all
three conditions with no change to the agent loops. `openai` is imported lazily so the package and
its tests never require the SDK unless you actually run a local model.

Caveat, stated honestly: 7-8B models are flaky at multi-step tool calling and clean structured
output. `respond` therefore tolerates a model that answers in prose instead of emitting a tool call
(the ReAct loop already nudges it back), and a tiny default trace set keeps a full run tractable on
a laptop. Expect lower, and noisier, results than Claude — that contrast *is* the finding.

    # one-time, in another terminal:
    #   brew install ollama && ollama serve
    #   ollama pull qwen2.5:7b-instruct
    from trace_reasoner.agent.local_client import LocalClient
    llm = LocalClient(model="qwen2.5:7b-instruct")   # default base_url is Ollama's
"""

from __future__ import annotations

import json

from trace_reasoner.agent.llm import (
    AssistantTurn,
    Message,
    ToolCall,
    ToolResultsMessage,
    ToolSpec,
    Usage,
    UserMessage,
)

# Ollama's OpenAI-compatible endpoint. Override base_url for llama.cpp (--api), LM Studio, vLLM, etc.
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


class LocalClient:
    """An LLMClient that calls a local OpenAI-compatible server (Ollama by default).

    Same contract as AnthropicClient: `respond(system, messages, tools) -> AssistantTurn`. The
    model name is the server's tag (e.g. "qwen2.5:7b-instruct", "llama3.1:8b-instruct-q4_0").
    `api_key` is ignored by local servers but the SDK requires a non-empty string.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client=None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if client is None:
            from openai import OpenAI  # lazy: only needed for local runs

            client = OpenAI(base_url=base_url, api_key="local-no-key-needed")
        self._client = client

    def respond(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        api_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
        api_messages = [{"role": "system", "content": system}]
        for m in messages:
            api_messages.extend(self._to_api(m))

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=api_tools,
            messages=api_messages,
        )
        choice = response.choices[0].message

        tool_calls = []
        for tc in choice.tool_calls or []:
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_args(tc.function.arguments))
            )
        usage = Usage(
            input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        )
        return AssistantTurn(text=choice.content or "", tool_calls=tool_calls, usage=usage)

    @staticmethod
    def _to_api(message: Message) -> list[dict]:
        """Convert one provider-agnostic transcript entry to OpenAI chat-completions message(s)."""
        if isinstance(message, UserMessage):
            return [{"role": "user", "content": message.text}]

        if isinstance(message, AssistantTurn):
            out: dict = {"role": "assistant", "content": message.text or None}
            if message.tool_calls:
                out["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in message.tool_calls
                ]
            return [out]

        if isinstance(message, ToolResultsMessage):
            # OpenAI wants one message per tool result, keyed by the originating call id.
            return [
                {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
                for r in message.results
            ]

        raise TypeError(f"unknown message type {type(message)!r}")


def _parse_args(raw) -> dict:
    """Tool-call arguments come back as a JSON string; small models sometimes emit junk.

    Tolerate a dict (some servers pre-parse), valid JSON, or unparseable output (return {} so the
    dispatcher reports a clean 'bad arguments' tool error rather than crashing the loop).
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
