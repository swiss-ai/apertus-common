"""Deep immutability: frozen models and validated container storage."""

import math
from collections.abc import Mapping, Sequence

import pytest
from pydantic import ValidationError

from apertus_common import Conversation, Harness, ToolCall, Wait


def test_frozen_assignment_is_rejected() -> None:
    call = ToolCall(id="f:1", name="f", args={})
    with pytest.raises(ValidationError):
        call.id = "g:2"


def test_mapping_storage_is_immutable() -> None:
    message = Harness(payload="x", extras={"lang": "en"})
    with pytest.raises(TypeError):
        message.extras["other"] = "de"
    with pytest.raises(TypeError):
        dict.__setitem__(message.extras, "other", "de")
    with pytest.raises(TypeError):
        message.extras.__init__({"other": "de"})


def test_sequence_storage_is_immutable() -> None:
    conversation = Conversation(items=[Harness(payload="h"), Wait()])
    with pytest.raises(AttributeError):
        conversation.items.append(Wait())
    with pytest.raises(TypeError):
        list.append(conversation.items, Wait())
    with pytest.raises(TypeError):
        conversation.items.__init__([Wait()])
    assert isinstance(conversation.items[0], Harness)
    assert conversation.items[0:1] == [conversation.items[0]]


def test_tool_call_args_are_recursively_immutable() -> None:
    call = ToolCall(id="f:1", name="f", args={"nested": []})
    assert isinstance(call.args, Mapping)
    with pytest.raises(TypeError):
        call.args["other"] = math.inf
    nested = call.args["nested"]
    assert isinstance(nested, Sequence)
    with pytest.raises(AttributeError):
        nested.append(math.inf)
