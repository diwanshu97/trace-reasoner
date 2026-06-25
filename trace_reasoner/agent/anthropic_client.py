"""LLMClient backed by Claude (Anthropic SDK).

Implements a single model turn — the ReAct loop in react.py owns the agentic
loop (manual loop pattern), which is what lets us bound steps, instrument token
usage, and stay model-agnostic. `anthropic` is imported lazily so the package
and its tests never require the SDK or an API key unless you actually run live.

Default model is claude-opus-4-8. Over a 200-trace eval that is not cheap; pass
model="claude-sonnet-4-6" (or haiku) to trade some quality for cost — your call,
not a silent downgrade.
"""

from __future__ import annotations

from trace_reasoner.agent.llm import (
    AssistantTurn,
    Message,
    ToolCall,
    ToolResultsMessage,
    ToolSpec,
    Usage,
    UserMessage,
)


class AnthropicClient:
    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 4096, client=None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is None:
            import anthropic  # lazy: only needed for live runs

            client = anthropic.Anthropic()
        self._client = client

    def respond(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        api_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=api_tools,
            messages=[self._to_api(m) for m in messages],
        )

        text = "".join(b.text for b in response.content if b.type == "text")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return AssistantTurn(text=text, tool_calls=tool_calls, usage=usage)

    @staticmethod
    def _to_api(message: Message) -> dict:
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.text}

        if isinstance(message, AssistantTurn):
            content: list[dict] = []
            if message.text:
                content.append({"type": "text", "text": message.text})
            for call in message.tool_calls:
                content.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            return {"role": "assistant", "content": content}

        if isinstance(message, ToolResultsMessage):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_call_id,
                        "content": r.content,
                        "is_error": r.is_error,
                    }
                    for r in message.results
                ],
            }

        raise TypeError(f"unknown message type {type(message)!r}")
