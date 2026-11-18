"""Caller mutation cannot alter validated native objects."""

import pytest
from pydantic import ValidationError

from apertus_common import Conversation, ToolSpec, User


def test_recursive_schema_freezing_and_serialization():
    source = {"type": "object", "properties": {"x": {"enum": [1, 2]}}}
    tool = ToolSpec(name="t", description="d", json_schema=source)
    source["properties"]["x"]["enum"].append(3)
    assert list(tool.json_schema["properties"]["x"]["enum"]) == [1, 2]
    with pytest.raises(TypeError):
        tool.json_schema["new"] = 1
    with pytest.raises(TypeError):
        tool.json_schema["properties"]["x"]["enum"][0] = 3
    with pytest.raises(TypeError):
        del tool.json_schema._FrozenDict__items


def test_conversation_copies_input_container_and_freezes_models():
    items = [User(payload="a")]
    conversation = Conversation(items=items)
    items.append(User(payload="b"))
    assert len(conversation.items) == 1
    with pytest.raises(ValidationError):
        conversation.items[0].payload = "changed"
    with pytest.raises(TypeError):
        del conversation.items._FrozenList__items
