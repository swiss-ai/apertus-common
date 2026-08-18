"""The message cast: per-class validation for input and output types."""

import math
from decimal import Decimal

import pytest
from pydantic import ValidationError

from apertus_common import (
    Attachment,
    GenericInput,
    GenericOutput,
    Harness,
    InputMessage,
    OutputMessage,
    Retrieval,
    ToolCall,
    ToolResult,
    ToolSpec,
)


def test_typing_bases_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError, match="typing base"):
        InputMessage()
    with pytest.raises(ValidationError, match="typing base"):
        OutputMessage()


def test_type_literals_are_fixed() -> None:
    assert Harness(payload="x").type == "harness"
    with pytest.raises(ValidationError):
        Harness(type="user", payload="x")  # type: ignore[arg-type]


def test_required_fields() -> None:
    with pytest.raises(ValidationError):
        ToolResult(payload="ok")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Retrieval(payload="x")  # type: ignore[call-arg]


def test_typed_header_fields_follow_grammar() -> None:
    assert Retrieval(src="internal-docs:7", payload="x").src == "internal-docs:7"
    ok = Attachment(name="report.pdf", mime="application/pdf", payload="Q2...")
    assert ok.name == "report.pdf"
    with pytest.raises(ValidationError):
        Retrieval(src="two words", payload="x")
    with pytest.raises(ValidationError):  # reject, never escape
        Attachment(name="my report.pdf", payload="x")


@pytest.mark.parametrize(
    ("direction", "type_value", "valid"),
    [
        ("in", "calendar_invite", True),
        # "harness" is a known *input* type; as an output it is open
        # vocabulary (the spec names it as a possible future output type).
        ("out", "harness", True),
        ("in", "user", False),
        ("out", "assistant", False),
        ("in", "bad type", False),
    ],
)
def test_generic_types(direction: str, type_value: str, valid: bool) -> None:
    cls = GenericInput if direction == "in" else GenericOutput
    if valid:
        assert cls(type=type_value, payload="x").type == type_value
    else:
        with pytest.raises(ValidationError):
            cls(type=type_value, payload="x")


@pytest.mark.parametrize("cls", [ToolCall, ToolSpec])
@pytest.mark.parametrize("name", ["bad name", "bad\n"])
def test_tool_name_grammar(cls: type, name: str) -> None:
    with pytest.raises(ValidationError):
        if cls is ToolCall:
            ToolCall(id="f:1", name=name, args={})
        else:
            ToolSpec(name=name, json_schema={})


def test_tool_call_args_accept_any_json_value() -> None:
    call = ToolCall(
        id="get_weather:4",
        name="get_weather",
        args={"city": "Lisbon", "units": ["C"], "detail": {"wind": True}},
    )
    assert call.args == {"city": "Lisbon", "units": ["C"], "detail": {"wind": True}}
    assert ToolCall(id="f:1", name="f", args=42).args == 42
    assert ToolCall(id="f:2", name="f", args=None).args is None


@pytest.mark.parametrize(
    "args",
    [
        {"nested": [math.nan]},
        {"nested": [math.inf]},
        Decimal("1.2"),
        b"x",
        {"nested": b"x"},
        {b"key": "value"},
    ],
)
def test_tool_call_args_rejected(args: object) -> None:
    with pytest.raises(ValidationError):
        ToolCall(id="f:1", name="f", args=args)


def test_tool_call_args_reject_json_number_overflow() -> None:
    with pytest.raises(ValidationError):
        ToolCall.model_validate_json('{"id":"f:1","name":"f","args":1e999}')
