"""Strict native storage boundaries and message dispatch."""

import json

import pytest
from pydantic import ValidationError

from apertus_common import (
    Conversation,
    Document,
    InputMessage,
    Message,
    OutputMessage,
    Reply,
    SystemPrompt,
    ToolSpec,
    User,
    Wait,
)
from conftest import full_cast


def test_full_native_roundtrip():
    conversation = full_cast()
    loaded = Conversation.from_json(conversation.to_json())
    assert loaded == conversation
    assert [type(item) for item in loaded.items] == [
        type(item) for item in conversation.items
    ]
    assert loaded.schema_version == 2


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"items":[]}',
        '{"schema_version":true}',
        '{"schema_version":3}',
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":1,"items":[{"type":"user","payload":"x"}]}',
    ],
)
def test_native_conversation_requires_version_and_discriminators(text):
    with pytest.raises(ValidationError):
        Conversation.from_json(text)


@pytest.mark.parametrize(
    "payload", ['{"type":"object","x":NaN}', '{"x":1e999}', '{"a":{"x":1,"x":2}}']
)
def test_strict_json_rejects_nonfinite_and_nested_duplicate_keys(payload):
    text = '{"name":"t","description":"d","schema":' + payload + "}"
    with pytest.raises(ValidationError):
        ToolSpec.from_json(text)


def test_message_dispatch_and_direction_boundaries():
    assert isinstance(Message.from_json(User(payload="hi").to_json()), User)
    assert isinstance(InputMessage.from_json(User(payload="hi").to_json()), User)
    assert isinstance(OutputMessage.from_json(Reply(payload="hi").to_json()), Reply)
    with pytest.raises(ValidationError):
        OutputMessage.from_json(User(payload="hi").to_json())
    with pytest.raises(ValidationError):
        User.from_json('{"payload":"hi"}')
    assert Wait.from_json('{"type":"wait"}') == Wait()


def test_aliases_are_native_storage_keys():
    data = json.loads(Document(payload="x", from_="user", id="a").to_json())
    assert data["from"] == "user"
    assert "from_" not in data


@pytest.mark.parametrize("loader", [Document, Message, InputMessage])
def test_native_document_loading_rejects_constructor_alias(loader):
    data = {"type": "document", "payload": "text", "from_": "user"}
    with pytest.raises(ValidationError):
        loader.from_json(json.dumps(data))
    assert Document(payload="text", from_="user").from_ == "user"


def test_native_tool_loading_rejects_constructor_alias():
    data = {"name": "f", "description": "tool", "json_schema": {}}
    with pytest.raises(ValidationError):
        ToolSpec.from_json(json.dumps(data))
    assert ToolSpec(name="f", description="tool", json_schema={}).json_schema == {}
    with pytest.raises(ValidationError):
        SystemPrompt.from_json(json.dumps({"thinking": "medium", "tools": [data]}))


@pytest.mark.parametrize(
    "data",
    [
        {
            "schema_version": 2,
            "items": [
                {
                    "type": "document",
                    "payload": "text",
                    "from_": "user",
                }
            ],
        },
        {
            "schema_version": 2,
            "system": {
                "thinking": "medium",
                "tools": [{"name": "f", "description": "tool", "json_schema": {}}],
            },
        },
    ],
)
def test_native_conversation_loading_rejects_nested_constructor_aliases(data):
    with pytest.raises(ValidationError):
        Conversation.from_json(json.dumps(data))


def test_native_json_schema_requires_storage_discriminators():
    schema = Conversation.model_json_schema()
    assert "schema_version" in schema["required"]
    assert {"type", "payload"} <= set(schema["$defs"]["User"]["required"])
    assert "type" in schema["$defs"]["Wait"]["required"]


@pytest.mark.parametrize(
    "message", [item for item in full_cast().items if isinstance(item, Message)]
)
def test_direction_is_derived_and_never_stored(message):
    expected = "in" if isinstance(message, InputMessage) else "out"
    assert message.direction == expected
    assert "direction" not in type(message).model_fields
    assert "direction" not in message.model_dump()
    assert "direction" not in type(message).model_json_schema()["properties"]
    assert Message.from_json(message.to_json()) == message
    with pytest.raises((AttributeError, ValidationError)):
        message.direction = expected
    old = {**json.loads(message.to_json()), "direction": expected}
    with pytest.raises(ValidationError, match="direction"):
        Message.from_json(json.dumps(old))
    with pytest.raises(ValidationError, match="direction"):
        type(message)(**old)
    with pytest.raises(ValidationError, match="direction"):
        Conversation.from_json(json.dumps({"schema_version": 2, "items": [old]}))


@pytest.mark.parametrize("name", ["memo", "user", "wait", "unknown_output", "system"])
def test_unknown_output_wrapper_preserves_wire_type(name):
    from jsonschema import Draft202012Validator

    from apertus_common import (
        AdapterError,
        MessageStart,
        TextStreamParser,
        UnknownOutput,
        render_str,
        to_openai,
        to_openai_output,
    )

    message = (
        UnknownOutput.builder()
        .output_type(name)
        .headers({"key": "value"})
        .payload("hé🙂")
        .build()
    )
    assert message.type == "unknown_output"
    assert message.output_type == name
    assert message.direction == "out"
    expected = {
        "type": "unknown_output",
        "output_type": name,
        "headers": {"key": "value"},
        "payload": "hé🙂",
    }
    assert json.loads(message.to_json()) == expected
    assert UnknownOutput.from_json(message.to_json()) == message
    assert OutputMessage.from_json(message.to_json()) == message
    with pytest.raises(ValidationError):
        InputMessage.from_json(message.to_json())
    conversation = Conversation(items=[message, Wait()])
    assert Conversation.from_json(conversation.to_json()) == conversation
    assert Draft202012Validator(Conversation.model_json_schema()).is_valid(
        json.loads(conversation.to_json())
    )
    wire = f'<|out|>{name}{{"key":"value"}}<|hdr|>hé🙂<|/out|><|wait|>'
    assert render_str(conversation) == wire
    parser = TextStreamParser()
    parser.push_text(wire)
    assert parser.finish() == conversation
    assert parser.take_events()[0] == MessageStart(0, name, {"key": "value"})
    report = message.check()
    assert report.errors[0].rule == "profile/known-output"
    assert repr(name) in report.errors[0].message
    for project in (to_openai, to_openai_output):
        with pytest.raises(AdapterError, match=f"unsupported output type {name!r}"):
            project(conversation)


@pytest.mark.parametrize(
    "item",
    [
        {"type": "memo", "payload": "x"},
        {"type": "unknown_output", "payload": "x"},
        {"type": "unknown_output", "output_type": "reply", "payload": "x"},
        {"type": "unknown_output", "output_type": "two words", "payload": "x"},
        {"type": "unknown_output", "output_type": 42, "payload": "x"},
        {"type": "user", "output_type": "memo", "payload": "x"},
        {"type": "wait", "direction": "out"},
    ],
)
def test_invalid_native_discriminators_match_schema(item):
    from jsonschema import Draft202012Validator

    data = {"schema_version": 2, "items": [item]}
    with pytest.raises(ValidationError):
        Conversation.from_json(json.dumps(data))
    assert not Draft202012Validator(Conversation.model_json_schema()).is_valid(data)
