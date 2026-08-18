"""Tool-call ids, payload parts, and payload coercion."""

import pytest
from pydantic import ValidationError

from apertus_common import (
    AudioPart,
    ImagePart,
    TextPart,
    ToolResult,
    User,
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


def test_image_geometry_both_or_neither() -> None:
    assert ImagePart().height is None
    assert ImagePart(height=2, width=3).width == 3
    with pytest.raises(ValidationError):
        ImagePart(height=2)
    with pytest.raises(ValidationError):
        ImagePart(width=3)
    with pytest.raises(ValidationError):
        ImagePart(height=0, width=3)
    with pytest.raises(ValidationError):
        ImagePart(height=True, width=True)
    with pytest.raises(ValidationError):
        ImagePart(height="2", width="3")
    with pytest.raises(ValidationError):
        ImagePart(height=b"2", width=b"3")
    assert ImagePart(height=2.0, width=3.0) == ImagePart(height=2, width=3)


def test_payload_parts() -> None:
    assert User(payload="hi").payload == [TextPart(text="hi")]
    parts = [
        TextPart(text="look: "),
        ImagePart(ref="a.jpg", height=2, width=3),
        AudioPart(ref="b.ogg"),
    ]
    assert User(payload=parts).payload == parts


def test_lone_surrogate_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        User(payload="\ud800")
