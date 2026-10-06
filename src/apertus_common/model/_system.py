"""System configuration and tool declarations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BeforeValidator, Field

from apertus_common.model._base import (
    HeaderValue,
    ImmutableList,
    Model,
    UnicodeScalarString,
)
from apertus_common.model._primitives import FiniteJsonObject, Thinking

if TYPE_CHECKING:
    from apertus_common.checker import CheckMode, CheckOptions


def _thinking(value: object) -> object:
    if not isinstance(value, str):
        raise ValueError("thinking must be a string or Thinking value")  # noqa: TRY004
    return value


class ToolSpec(Model, frozen=True):
    name: HeaderValue
    description: UnicodeScalarString
    json_schema: FiniteJsonObject = Field(
        validation_alias="schema", serialization_alias="schema"
    )
    policy: UnicodeScalarString | None = None


class SystemPrompt(Model, frozen=True):
    identity: UnicodeScalarString | None = None
    thinking: Annotated[Thinking, BeforeValidator(_thinking)]
    behavior: UnicodeScalarString | None = None
    tools: ImmutableList[ToolSpec] = Field(default_factory=tuple)
    environment: UnicodeScalarString | None = None

    @classmethod
    def build(
        cls,
        *,
        thinking: Thinking | str = Thinking.MEDIUM,
        identity: str | None = None,
        behavior: str | None = None,
        tools: Sequence[ToolSpec] = (),
        environment: str | None = None,
        check: CheckMode | CheckOptions | None = None,
    ) -> Self:
        from apertus_common.model._base import apply_model_check

        result = cls.model_validate(
            {
                "thinking": thinking,
                "identity": identity,
                "behavior": behavior,
                "tools": tools,
                "environment": environment,
            }
        )
        return apply_model_check(result, check)
