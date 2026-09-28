"""Tool-call ids, payload parts, and payload coercion."""

import json

import pytest
from pydantic import ValidationError

from apertus_common import (
    TextPart,
    ToolResult,
    User,
    from_json,
    is_canonical_id,
)


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("get_weather:0", True),
        ("call_abc123", True),
        ("a", True),
        ("A-b.c:2", True),
        ("a" * 64, True),
        ("", False),
        ("a" * 65, False),
        ("bad id", False),
        ("id;1", False),
        ("a|b", False),
        ("call:1\n", False),
    ],
)
def test_tool_call_id_shape(value: str, valid: bool) -> None:
    if valid:
        assert ToolResult(id=value, payload="ok").id == value
    else:
        with pytest.raises(ValidationError):
            ToolResult(id=value, payload="ok")


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("get_weather:0", True),
        ("bash:57", True),
        ("Bash_2:007", True),
        ("call_abc123", False),
        ("a:b", False),
        ("a.b:1", False),
        ("bash:1\n", False),
    ],
)
def test_is_canonical_id(value: str, canonical: bool) -> None:
    assert is_canonical_id(value) is canonical


def test_payload_parts() -> None:
    assert User(payload="hi").payload == [TextPart(text="hi")]
    parts = [
        TextPart(text="look: "),
        TextPart(text="a text attachment"),
    ]
    assert User(payload=parts).payload == parts


def test_payload_mapping_requires_text_tag() -> None:
    with pytest.raises(ValidationError):
        User.model_validate({"payload": [{"text": "missing part tag"}]})


def test_lone_surrogate_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        User(payload="\ud800")


@pytest.mark.parametrize("kind", ["image", "audio"])
@pytest.mark.parametrize("message_type", ["user", "attachment"])
def test_native_media_parts_are_rejected(kind: str, message_type: str) -> None:
    data = {
        "items": [
            {
                "direction": "in",
                "type": message_type,
                "payload": [{"part": kind, "ref": "asset://example"}],
            }
        ]
    }
    with pytest.raises(ValidationError):
        from_json(json.dumps(data))
