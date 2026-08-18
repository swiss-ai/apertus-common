"""Executable copy of the format spec's order/authority table."""

from typing import Literal

import pytest

from apertus_common import (
    Assistant,
    Attachment,
    Conversation,
    Document,
    Event,
    GenericInput,
    GenericOutput,
    Harness,
    InputMessage,
    OutputMessage,
    Retrieval,
    Think,
    ToolCall,
    ToolResult,
    User,
    VerifiableAnswer,
    type_info,
)

# (type, direction, order, authority): literal rows from the format
# spec's message tables.
SPEC_TABLE: list[tuple[str, str, int | None, int | None]] = [
    ("harness", "in", 1, 2),
    ("tool_result", "in", 2, 4),
    ("retrieval", "in", 3, 4),
    ("event", "in", 3, 4),
    ("attachment", "in", 4, 4),
    ("user", "in", 5, 3),
    ("document", "in", None, 4),
    ("assistant", "out", None, None),
    ("think", "out", None, None),
    ("tool_call", "out", None, None),
    ("verifiable_answer", "out", None, None),
]

CAST: dict[type[InputMessage] | type[OutputMessage], tuple[str, str]] = {
    Harness: ("in", "harness"),
    ToolResult: ("in", "tool_result"),
    Retrieval: ("in", "retrieval"),
    Event: ("in", "event"),
    Attachment: ("in", "attachment"),
    User: ("in", "user"),
    Document: ("in", "document"),
    Assistant: ("out", "assistant"),
    Think: ("out", "think"),
    ToolCall: ("out", "tool_call"),
    VerifiableAnswer: ("out", "verifiable_answer"),
}


@pytest.mark.parametrize(("type_name", "direction", "order", "authority"), SPEC_TABLE)
def test_registry_matches_spec_table(
    type_name: str, direction: str, order: int | None, authority: int | None
) -> None:
    info = type_info(direction, type_name)  # type: ignore[arg-type]
    assert info.type == type_name
    assert info.direction == direction
    assert info.order == order
    assert info.authority == authority


def test_class_labels_match_registry() -> None:
    for message_class, (direction, type_name) in CAST.items():
        info = type_info(direction, type_name)  # type: ignore[arg-type]
        assert message_class.order == info.order
        assert message_class.authority == info.authority


def test_open_vocabulary_defaults() -> None:
    unknown_in = type_info("in", "totally_custom")
    assert (unknown_in.order, unknown_in.authority) == (3, 4)
    unknown_out = type_info("out", "totally_custom")
    assert (unknown_out.order, unknown_out.authority) == (None, None)
    assert GenericInput.order == 3
    assert GenericInput.authority == 4
    assert GenericOutput.order is None
    assert GenericOutput.authority is None


def test_invalid_direction_is_rejected_at_runtime() -> None:
    with pytest.raises(ValueError):
        type_info("sideways", "custom")  # type: ignore[arg-type]


def test_external_subclass_does_not_poison_open_vocabulary() -> None:
    class CustomInput(InputMessage):
        type: Literal["custom"] = "custom"
        payload: str

    dumped = CustomInput(payload="x").model_dump()
    generic = GenericInput(type="custom", payload="x")
    restored = Conversation.model_validate({"items": [dumped]})

    assert type_info("in", "custom").order == 3
    assert restored.items == [generic]
