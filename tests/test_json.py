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
    data = {"direction": "in", "type": "document", "payload": "text", "from_": "user"}
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
            "schema_version": 1,
            "items": [
                {
                    "direction": "in",
                    "type": "document",
                    "payload": "text",
                    "from_": "user",
                }
            ],
        },
        {
            "schema_version": 1,
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
    assert {"direction", "type", "payload"} <= set(schema["$defs"]["User"]["required"])
    assert "type" in schema["$defs"]["Wait"]["required"]
