"""Streaming projections share batch semantics and fail terminally."""

from typing import Any

import pytest

from apertus_common.errors import AdapterError
from apertus_common.formats import MappingPolicy, OpenAIStreamMapper, to_openai_output
from apertus_common.model import Call, Claim, Conversation, Reply, Think, Wait
from apertus_common.parsing import (
    MessageCompleted,
    MessageStart,
    PayloadDelta,
    WaitReached,
)


def events(message: Any, index: int = 0) -> list[Any]:
    headers = (
        {"name": message.name, "counter": message.counter}
        if isinstance(message, Call)
        else {}
    )
    return [
        MessageStart(index, message.type, headers),
        *[PayloadDelta(index, char) for char in message.payload],
        MessageCompleted(index, message),
    ]


def test_stream_batch_parity_with_unicode_and_invalid_tool_json() -> None:
    messages = [
        Think(payload="Réfléchir 🦉"),
        Reply(payload="Checking"),
        Call(name="weather", counter=0, payload='{"city":'),
        Call(name="weather", counter=1, payload="[]"),
    ]
    mapper = OpenAIStreamMapper()
    deltas = []
    for index, message in enumerate(messages):
        for event in events(message, index):
            deltas.extend(mapper.push(event))
    mapper.push(WaitReached())
    result = mapper.finish()
    assert result == to_openai_output(Conversation(items=[*messages, Wait()]))
    assert (
        "".join(delta.get("reasoning_content", "") for delta in deltas)
        == messages[0].payload
    )
    assert "".join(delta.get("content", "") for delta in deltas) == messages[1].payload
    for index, call in enumerate(messages[2:]):
        parts = [
            tool["function"].get("arguments", "")
            for delta in deltas
            for tool in delta.get("tool_calls", [])
            if tool["index"] == index
        ]
        assert "".join(parts) == call.payload
    assert mapper.finish() is result
    with pytest.raises(AdapterError):
        mapper.push(WaitReached())


def test_loss_diagnostics_available_before_finish_and_retained_in_result() -> None:
    policy = MappingPolicy(merge_assistant_messages=True, omit_claims=True)
    mapper = OpenAIStreamMapper(policy=policy)
    messages = [
        Reply(payload="one"),
        Claim(payload="omit"),
        Think(payload="late"),
        Reply(payload="two"),
    ]
    drained = []
    for index, message in enumerate(messages):
        for event in events(message, index):
            mapper.push(event)
            drained.extend(mapper.take_diagnostics())
    assert drained and all(d.lossy for d in drained)
    assert mapper.take_diagnostics() == []
    result = mapper.finish()
    assert tuple(drained) == result.diagnostics
    assert result == to_openai_output(Conversation(items=messages), policy=policy)
    assert result.value["content"] == "onetwo"


@pytest.mark.parametrize(
    "invalid_events",
    [
        [PayloadDelta(0, "orphan")],
        [MessageCompleted(0, Reply(payload="orphan"))],
        [MessageStart(1, "reply", {})],
        [MessageStart(0, "reply", {}), MessageStart(0, "reply", {})],
        [MessageStart(0, "reply", {}), PayloadDelta(1, "wrong index")],
        [MessageStart(0, "reply", {}), WaitReached()],
        [
            MessageStart(0, "reply", {}),
            PayloadDelta(0, "a"),
            MessageCompleted(0, Reply(payload="b")),
        ],
        [MessageStart(0, "reply", {}), MessageCompleted(0, Think(payload=""))],
        [MessageStart(0, "call", {"name": "weather", "counter": True})],
        [MessageStart(0, "call", {"name": "weather", "counter": -1})],
        [
            MessageStart(0, "call", {"name": "weather", "counter": 0}),
            MessageCompleted(0, Call(name="other", counter=0, payload="")),
        ],
        [*events(Reply(payload="a")), MessageStart(0, "reply", {})],
        [
            *events(Call(name="weather", counter=0, payload="")),
            MessageStart(1, "call", {"name": "weather", "counter": 0}),
        ],
        [WaitReached(), MessageStart(0, "reply", {})],
    ],
)
def test_bad_event_sequences_fail_terminally(invalid_events: list[Any]) -> None:
    mapper = OpenAIStreamMapper()
    with pytest.raises(AdapterError):
        for event in invalid_events:
            mapper.push(event)
    with pytest.raises(AdapterError):
        mapper.finish()
    with pytest.raises(AdapterError):
        mapper.push(WaitReached())


def test_unfinished_message_finish_is_terminal() -> None:
    mapper = OpenAIStreamMapper()
    mapper.push(MessageStart(0, "reply", {}))
    with pytest.raises(AdapterError):
        mapper.finish()
    with pytest.raises(AdapterError):
        mapper.push(MessageCompleted(0, Reply(payload="")))


@pytest.mark.parametrize(
    "start",
    [
        MessageStart(0, "reply", {"ignored": "extra"}),
        MessageStart(0, "call", {"name": "weather", "counter": 0, "ignored": "extra"}),
        MessageStart(0, "call", {"name": "", "counter": 0}),
    ],
)
def test_invalid_event_headers_raise_adapter_error_and_poison_mapper(
    start: MessageStart,
) -> None:
    mapper = OpenAIStreamMapper()
    with pytest.raises(AdapterError):
        mapper.push(start)
    with pytest.raises(AdapterError):
        mapper.finish()


def test_parser_events_project_to_same_result_at_every_chunk_boundary() -> None:
    from apertus_common.core import render_str
    from apertus_common.parsing import TextStreamParser

    conversation = Conversation(
        items=[
            Think(payload="Réfléchir 🦉"),
            Reply(payload=""),
            Call(name="weather", counter=0, payload='{"broken":'),
            Wait(),
        ]
    )
    wire = render_str(conversation)
    expected = to_openai_output(conversation)
    for split in range(len(wire) + 1):
        parser = TextStreamParser()
        mapper = OpenAIStreamMapper()
        for chunk in (wire[:split], wire[split:]):
            if not chunk:
                continue
            parser.push_text(chunk)
            for event in parser.take_events():
                mapper.push(event)
        assert parser.finish() == conversation
        assert mapper.finish() == expected
