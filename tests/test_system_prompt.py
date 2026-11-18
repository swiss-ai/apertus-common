"""Authoring defaults never leak into native system loading."""

import pytest
from pydantic import ValidationError

from apertus_common import SystemPrompt, Thinking, ToolSpec


def test_thinking_required_directly_but_authoring_defaults_medium():
    with pytest.raises(ValidationError):
        SystemPrompt()
    with pytest.raises(ValidationError):
        SystemPrompt.from_json("{}")
    assert SystemPrompt.build().thinking is Thinking.MEDIUM
    assert SystemPrompt.builder().build().thinking is Thinking.MEDIUM
    assert SystemPrompt(thinking="high").thinking is Thinking.HIGH
    for value in (1, True, "max"):
        with pytest.raises(ValidationError):
            SystemPrompt(thinking=value)


def test_tool_declaration_requires_description_and_preserves_schema():
    tool = ToolSpec(
        name="tool", description="Description", json_schema={"type": "object"}
    )
    assert '"schema":' in tool.to_json()
    assert ToolSpec.from_json(tool.to_json()) == tool
    assert SystemPrompt.builder().tool(tool).build().tools == [tool]
    with pytest.raises(ValidationError):
        ToolSpec(name="tool", json_schema={})
