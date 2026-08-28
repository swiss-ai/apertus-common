"""formats: the OpenAI chat-completions mapping (from_openai / to_openai)."""

import pickle
from typing import Any

import pytest

from apertus_common import (
    AdapterError,
    Assistant,
    AudioPart,
    ConformanceError,
    Conversation,
    Effort,
    Harness,
    ImagePart,
    Retrieval,
    SystemPrompt,
    TextPart,
    Think,
    ToolCall,
    ToolResult,
    ToolSpec,
    User,
    VerifiableAnswer,
    Wait,
    check,
    from_openai,
    to_openai,
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "One city per call.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
REQUEST: list[dict[str, Any]] = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Lisbon or Porto?"},
    {
        "role": "assistant",
        "reasoning_content": "Fetch both.",
        "content": "Checking.",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Lisbon"}',
                },
            },
            {
                "id": "get_weather:1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Porto"}'},
            },
        ],
    },
    {"role": "tool", "tool_call_id": "call_abc123", "content": "24C"},
    {"role": "tool", "tool_call_id": "get_weather:1", "content": "19C"},
    {"role": "assistant", "content": "Lisbon is warmer."},
]


def test_representative_round_trip() -> None:
    conv = from_openai(REQUEST, TOOLS, effort="high")
    assert conv.system == SystemPrompt(
        behavior="Be brief.",
        effort=Effort.HIGH,
        tools=[
            ToolSpec(
                name="get_weather",
                json_schema=TOOLS[0]["function"]["parameters"],
                policy="One city per call.",
            )
        ],
    )
    assert list(conv.items) == [
        User(payload="Lisbon or Porto?"),
        Think(payload="Fetch both."),
        Assistant(payload="Checking."),
        ToolCall(id="call_abc123", name="get_weather", args={"city": "Lisbon"}),
        ToolCall(id="get_weather:1", name="get_weather", args={"city": "Porto"}),
        Wait(),
        ToolResult(id="call_abc123", payload="24C"),
        ToolResult(id="get_weather:1", payload="19C"),
        Assistant(payload="Lisbon is warmer."),
        Wait(),
    ]
    assert check(conv).ok
    assert to_openai(conv) == {
        "messages": REQUEST,
        "tools": TOOLS,
        "reasoning_effort": "high",
    }


def test_wait_closes_each_assistant_run_and_prefill_leaves_it_open() -> None:
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "assistant", "content": "c"},
        {"role": "user", "content": "d"},
        {"role": "assistant", "content": "e"},
    ]
    closed = from_openai(messages)
    assert [type(i) for i in closed.items] == [
        User,
        Assistant,
        Assistant,
        Wait,
        User,
        Assistant,
        Wait,
    ]
    assert closed.system == SystemPrompt(effort=Effort.MEDIUM)
    prefill = from_openai(messages, continue_final_message=True)
    assert not isinstance(prefill.items[-1], Wait)
    assert [v.rule for v in check(prefill).warnings] == ["wait/unterminated-transcript"]
    # to_openai: consecutive Assistant items become separate messages, no Wait
    assert to_openai(prefill)["messages"] == messages


def test_later_system_is_harness_and_back() -> None:
    messages = [
        {"role": "developer", "content": "Rules."},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "Time is 12:00."},
    ]
    conv = from_openai(messages, normalize=False)
    assert conv.system is not None and conv.system.behavior == "Rules."
    assert from_openai([{"role": "system", "content": ""}]).system == SystemPrompt(
        effort=Effort.MEDIUM
    )
    assert list(conv.items) == [
        User(payload="hi"),
        Harness(payload="Time is 12:00."),
    ]
    assert to_openai(conv)["messages"] == [
        {"role": "system", "content": "Rules."},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "Time is 12:00."},
    ]


def test_multimodal_parts_both_ways() -> None:
    parts = [
        {"type": "text", "text": "Look: "},
        {"type": "image_url", "image_url": {"url": "https://x/1.png"}},
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
    ]
    conv = from_openai([{"role": "user", "content": parts}])
    assert list(conv.items[0].payload) == [  # type: ignore[union-attr]
        TextPart(text="Look: "),
        ImagePart(ref="https://x/1.png"),
        AudioPart(ref="AAAA"),
    ]
    out = to_openai(conv)["messages"][0]["content"]
    assert out == [
        {"type": "text", "text": "Look: "},
        {"type": "image_url", "image_url": {"url": "https://x/1.png"}},
        {"type": "input_audio", "input_audio": {"data": "AAAA"}},  # format dropped
    ]
    single = from_openai([{"role": "user", "content": [{"type": "text", "text": "t"}]}])
    assert to_openai(single)["messages"][0]["content"] == "t"


def test_tool_arguments_parsing() -> None:
    def call(arguments: object) -> dict[str, Any]:
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "f:1",
                    "type": "function",
                    "function": {"name": "f", "arguments": arguments},
                }
            ],
        }

    assert from_openai([call("")]).items[0] == ToolCall(id="f:1", name="f", args={})
    assert from_openai([call({"x": 1})]).items[0] == ToolCall(
        id="f:1", name="f", args={"x": 1}
    )
    with pytest.raises(AdapterError) as info:
        from_openai([{"role": "user", "content": "q"}, call("{not json")])
    assert (info.value.where, info.value.index) == ("message", 1)
    assert isinstance(info.value.__cause__, ValueError)
    assert str(info.value).startswith("openai message 1: ")


def test_reasoning_never_enters_content() -> None:
    conv = Conversation(
        items=[
            Assistant(payload="b"),
            Think(payload="a"),
            Assistant(payload="d"),
            ToolCall(id="f:1", name="f", args={}),
            Wait(),
        ]
    )
    assert to_openai(conv)["messages"] == [
        {"role": "assistant", "content": "b"},
        {
            "role": "assistant",
            "reasoning_content": "a",
            "content": "d",
            "tool_calls": [
                {
                    "id": "f:1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
    ]
    think_only = Conversation(items=[Think(payload="a"), Wait()])
    assert to_openai(think_only)["messages"] == [
        {"role": "assistant", "reasoning_content": "a", "content": None}
    ]


def test_tools_mapping_defaults() -> None:
    conv = from_openai([], [{"type": "function", "function": {"name": "f"}}])
    assert conv.system is not None
    assert list(conv.system.tools) == [
        ToolSpec(name="f", json_schema={"type": "object"})
    ]
    assert to_openai(conv)["tools"] == [
        {
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object"}},
        }
    ]


@pytest.mark.parametrize(
    ("messages", "index", "fragment"),
    [
        ([{"role": "critic", "content": "x"}], 0, "unsupported role"),
        (
            [{"role": "user", "content": "q"}, {"role": "tool", "content": "x"}],
            1,
            "tool_call_id",
        ),
        (
            [{"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]}],
            0,
            "needs id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "bad id", "function": {"name": "f", "arguments": "{}"}}
                    ],
                }
            ],
            0,
            "tool-call id",
        ),
        (
            [{"role": "user", "content": [{"type": "video"}]}],
            0,
            "unsupported content part",
        ),
    ],
)
def test_from_openai_errors(
    messages: list[dict[str, Any]], index: int, fragment: str
) -> None:
    with pytest.raises(AdapterError, match=fragment) as info:
        from_openai(messages)
    assert (info.value.where, info.value.index) == ("message", index)


def test_from_openai_rejects_bad_effort_and_tools() -> None:
    with pytest.raises(AdapterError, match="system prompt: ") as info:
        from_openai([], effort="extreme")
    assert (info.value.where, info.value.index) == ("system", -1)
    with pytest.raises(AdapterError, match="system prompt: tools\\[0\\]") as info:
        from_openai([], [{"type": "retrieval"}])
    assert info.value.where == "system"


@pytest.mark.parametrize(
    ("conversation", "where", "index", "fragment"),
    [
        (
            Conversation(items=[Retrieval(src="d:1", payload="x")]),
            "item",
            0,
            "'retrieval'",
        ),
        (
            Conversation(items=[User(payload="hi"), VerifiableAnswer(payload="42")]),
            "item",
            1,
            "'verifiable_answer'",
        ),
        (
            Conversation(items=[Harness(payload="x", extras={"k": "v"})]),
            "item",
            0,
            "extras",
        ),
        (
            Conversation(
                system=SystemPrompt(effort="low", extra_tags={"deploy": "pilot"})
            ),
            "system",
            -1,
            "extra_tags",
        ),
    ],
)
def test_to_openai_unmappable_raises(
    conversation: Conversation, where: str, index: int, fragment: str
) -> None:
    with pytest.raises(AdapterError, match=fragment) as info:
        to_openai(conversation)
    assert (info.value.where, info.value.index) == (where, index)
    restored = pickle.loads(pickle.dumps(info.value))
    assert str(restored) == str(info.value)


def test_system_prompt_collapses_to_one_string() -> None:
    conv = Conversation(
        system=SystemPrompt(
            identity="I am Aria.", behavior="Be brief.", environment="web", effort="low"
        )
    )
    assert to_openai(conv) == {
        "messages": [{"role": "system", "content": "I am Aria.\n\nBe brief.\n\nweb"}],
        "reasoning_effort": "low",
    }
    assert to_openai(Conversation()) == {"messages": []}


def test_ingest_flags() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "notice"},
    ]
    assert [type(i) for i in from_openai(messages).items] == [Harness, User]
    observed = from_openai(messages, normalize=False)
    assert [type(i) for i in observed.items] == [User, Harness]
    conv, report = from_openai(messages, normalize=False, check=True)
    assert conv == observed and not report.ok
    with pytest.raises(ConformanceError):
        from_openai(messages, normalize=False, strict=True)
    assert from_openai(messages, strict=True) == from_openai(messages)
