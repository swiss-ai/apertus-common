"""Conversations retain arrival order and author no implicit burst boundaries."""

import pytest
from pydantic import ValidationError

from apertus_common import Conversation, Host, Reply, SystemPrompt, User, Wait


def test_order_and_typed_item_identity_are_preserved():
    user, host = User(payload="u"), Host(payload="h")
    conversation = Conversation(items=[user, host, Reply(payload="r")])
    assert conversation.items[0] is user
    assert conversation.items[1] is host
    assert not any(isinstance(item, Wait) for item in conversation.items)


def test_builder_shortcuts_and_snapshot():
    builder = Conversation.builder().system(SystemPrompt.build()).user("hello")
    first = builder.build()
    second = builder.reply("hi").wait().build()
    assert len(first.items) == 1
    assert len(second.items) == 3
    assert isinstance(second.items[-1], Wait)
    with pytest.raises(ValueError, match="already assigned"):
        builder.system(SystemPrompt.build())


def test_system_cannot_be_an_item():
    with pytest.raises(ValidationError):
        Conversation(items=[SystemPrompt.build()])


def test_unknown_inputs_and_old_items_are_rejected():
    for item in (
        {"type": "future", "payload": ""},
        {"type": "assistant", "payload": "", "extras": {}},
    ):
        with pytest.raises(ValidationError):
            Conversation(items=[item])
