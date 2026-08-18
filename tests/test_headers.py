"""Header grammar: open vocabulary keys/values, collisions, error types."""

import pytest
from pydantic import ValidationError

from apertus_common import ApertusFormatError, Harness, HeaderError, ToolResult


def test_valid_extras() -> None:
    message = Harness(payload="x", extras={"lang": "en", "V2": "yes_1"})
    assert message.extras == {"lang": "en", "V2": "yes_1"}


@pytest.mark.parametrize("key", ["a b", "", "k=v", "käse", "a|b", "a-b", "trailing\n"])
def test_bad_keys_rejected(key: str) -> None:
    with pytest.raises(ValidationError):
        Harness(payload="x", extras={key: "v"})


@pytest.mark.parametrize(
    "value",
    ["", "a b", "a=b", "a\tb", "a\nb", "<|hdr|>", "x|>y", "a<|b", "\ud800"],
)
def test_bad_values_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        Harness(payload="x", extras={"k": value})


def test_extras_key_collisions() -> None:
    with pytest.raises(ValidationError):
        ToolResult(id="bash:1", payload="ok", extras={"id": "bash:2"})
    for key in ("type", "direction"):
        with pytest.raises(ValidationError):
            Harness(payload="x", extras={key: "user"})


def test_error_hierarchy() -> None:
    assert issubclass(HeaderError, ApertusFormatError)
    assert issubclass(HeaderError, ValueError)
    assert issubclass(ApertusFormatError, Exception)
    # header violations surface through pydantic's ValidationError
    with pytest.raises(ValidationError):
        Harness(payload="x", extras={"bad key": "v"})
