"""SystemPrompt: required effort, tool specs, extra tags."""

import pytest
from pydantic import ValidationError

from apertus_common import Effort, SystemPrompt, ToolSpec


def test_effort_field() -> None:
    assert [e.value for e in Effort] == ["low", "medium", "high"]
    assert SystemPrompt(effort="low").effort is Effort.LOW
    with pytest.raises(ValidationError):
        SystemPrompt()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        SystemPrompt(effort="extreme")
    with pytest.raises(ValidationError):
        SystemPrompt(effort=b"low")  # type: ignore[arg-type]


def test_tool_schema_requires_json_object() -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name="tool", json_schema={"value": {"not", "json"}})
    with pytest.raises(ValidationError):
        ToolSpec(name="tool", json_schema="not an object")  # type: ignore[arg-type]


@pytest.mark.parametrize("tag", ["identity", "1bad", "bad\n"])
def test_bad_extra_tags_rejected(tag: str) -> None:
    with pytest.raises(ValidationError):
        SystemPrompt(effort="low", extra_tags={tag: "x"})
