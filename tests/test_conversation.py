"""Conversation: round-trip class preservation, union edges, normalization."""

import pytest
from pydantic import ValidationError

from apertus_common import (
    Assistant,
    Conversation,
    Document,
    Event,
    GenericInput,
    Harness,
    Retrieval,
    ToolResult,
    User,
    Wait,
)
from conftest import full_cast, pretraining


@pytest.mark.parametrize("mode", ["json", "python"])
def test_round_trip_preserves_classes(mode: str) -> None:
    for conversation in (full_cast(), pretraining()):
        if mode == "json":
            restored = Conversation.model_validate_json(conversation.model_dump_json())
        else:
            restored = Conversation.model_validate(conversation.model_dump())
        assert restored == conversation
        for original, back in zip(conversation.items, restored.items, strict=True):
            assert type(original) is type(back)
    assert pretraining().system is None


@pytest.mark.parametrize(
    "item",
    [
        {"direction": "sideways", "type": "user", "payload": "x"},
        {"direction": "in", "payload": "x"},
        {"direction": "in", "type": 3, "payload": "x"},
        {"type": "wait", "extra": "field"},
        {"id": "f:1", "payload": "x"},
        {},
    ],
)
def test_union_rejects_malformed_items(item: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Conversation.model_validate({"items": [item]})


def test_normalized_sorts_a_boundary_run() -> None:
    conversation = Conversation(
        items=[
            User(payload="u"),
            GenericInput(type="note", payload="n"),
            ToolResult(id="bash:1", payload="r"),
            Document(payload="d"),
            Harness(payload="h"),
            Wait(),
        ]
    )
    normalized = conversation.normalized()
    # unlabeled types (GenericInput, Document) key at position 3, stably
    assert [type(i) for i in normalized.items] == [
        Harness,
        ToolResult,
        GenericInput,
        Document,
        User,
        Wait,
    ]
    # the original is untouched
    assert [type(i) for i in conversation.items] == [
        User,
        GenericInput,
        ToolResult,
        Document,
        Harness,
        Wait,
    ]


def test_normalized_is_stable_within_a_position() -> None:
    conversation = Conversation(
        items=[
            Event(payload="e1"),
            ToolResult(id="a:1", payload="1"),
            Retrieval(src="s:1", payload="r1"),
            Event(payload="e2"),
            ToolResult(id="a:2", payload="2"),
        ]
    )
    normalized = conversation.normalized()
    assert [type(i) for i in normalized.items] == [
        ToolResult,
        ToolResult,
        Event,
        Retrieval,
        Event,
    ]
    tool_results = [i for i in normalized.items if isinstance(i, ToolResult)]
    assert [t.id for t in tool_results] == ["a:1", "a:2"]
    rest = [i for i in normalized.items if not isinstance(i, ToolResult)]
    assert [r.payload for r in rest] == ["e1", "r1", "e2"]


def test_normalized_runs_are_independent_and_outputs_never_move() -> None:
    conversation = Conversation(
        items=[
            User(payload="u1"),
            Assistant(payload="a"),
            User(payload="u2"),
            Harness(payload="h"),
            Wait(),
            ToolResult(id="bash:1", payload="r"),
        ]
    )
    normalized = conversation.normalized()
    assert [type(i) for i in normalized.items] == [
        User,
        Assistant,
        Harness,
        User,
        Wait,
        ToolResult,
    ]


def test_normalized_is_idempotent_and_canonical_is_a_fixed_point() -> None:
    messy = Conversation(items=[User(payload="u"), Harness(payload="h"), Wait()])
    normalized = messy.normalized()
    assert normalized.normalized() == normalized
    assert normalized.normalized() is not normalized  # always a new container
    canonical = Conversation(items=[Harness(payload="h"), User(payload="u")])
    assert canonical.normalized() == canonical
