"""LocalClient wire-format translation, verified offline against a fake OpenAI client.

No server and no `openai` SDK call: we inject a stub that records the request and returns a canned
chat-completion, so the (project dataclasses <-> OpenAI chat format) conversion is tested in CI.
The live path against Ollama is exercised by run_local.py, not here.
"""

from types import SimpleNamespace

from trace_reasoner.agent.llm import (
    AssistantTurn,
    ToolCall,
    ToolResult,
    ToolResultsMessage,
    UserMessage,
)
from trace_reasoner.agent.local_client import LocalClient, _parse_args
from trace_reasoner.agent.tools_runtime import tool_specs


class FakeOpenAI:
    """Records the last create() request and returns a scripted message."""

    def __init__(self, message):
        self._message = message
        self.last_request = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_request = kwargs
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)], usage=usage)


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_respond_parses_a_tool_call():
    fake = FakeOpenAI(_msg(content="", tool_calls=[_tool_call("c1", "survey", "{}")]))
    llm = LocalClient(client=fake)
    turn = llm.respond("sys", [UserMessage("go")], tool_specs())

    assert isinstance(turn, AssistantTurn)
    assert [tc.name for tc in turn.tool_calls] == ["survey"]
    assert turn.tool_calls[0].arguments == {}
    assert turn.usage.input_tokens == 11 and turn.usage.output_tokens == 7


def test_respond_parses_arguments_json():
    args = '{"span_id": "s3", "direction": "children"}'
    fake = FakeOpenAI(_msg(tool_calls=[_tool_call("c2", "walk_tree", args)]))
    turn = LocalClient(client=fake).respond("sys", [UserMessage("go")], tool_specs())
    assert turn.tool_calls[0].arguments == {"span_id": "s3", "direction": "children"}


def test_respond_handles_prose_without_tool_calls():
    # Small models often answer in prose instead of calling a tool — must not crash.
    fake = FakeOpenAI(_msg(content="I think it is the payment span.", tool_calls=None))
    turn = LocalClient(client=fake).respond("sys", [UserMessage("go")], tool_specs())
    assert turn.tool_calls == []
    assert "payment" in turn.text


def test_request_shape_system_tools_and_roles():
    fake = FakeOpenAI(_msg(tool_calls=[_tool_call("c3", "survey", "{}")]))
    llm = LocalClient(model="qwen2.5:7b-instruct", client=fake)
    transcript = [
        UserMessage("localize trace t1"),
        AssistantTurn(text="", tool_calls=[ToolCall("c0", "survey", {})]),
        ToolResultsMessage([ToolResult("c0", '{"n_spans": 5}')]),
    ]
    llm.respond("SYSTEM", transcript, tool_specs())
    req = fake.last_request

    assert req["model"] == "qwen2.5:7b-instruct"
    assert req["messages"][0] == {"role": "system", "content": "SYSTEM"}
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    # the assistant turn carried its tool_call through, and the tool result is keyed back to it
    assert req["messages"][2]["tool_calls"][0]["id"] == "c0"
    assert req["messages"][3]["tool_call_id"] == "c0"
    # tools are advertised in OpenAI function form
    assert req["tools"][0]["type"] == "function"
    assert {t["function"]["name"] for t in req["tools"]} >= {"survey", "walk_tree", "submit_hypotheses"}


def test_parse_args_tolerates_garbage():
    assert _parse_args('{"a": 1}') == {"a": 1}
    assert _parse_args({"a": 1}) == {"a": 1}      # some servers pre-parse
    assert _parse_args("not json at all") == {}   # junk -> empty, dispatcher reports clean error
    assert _parse_args("") == {}
    assert _parse_args("[1, 2, 3]") == {}          # non-object JSON -> empty
