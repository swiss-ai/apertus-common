"""Typed system prompt, tool inventory, and custom-tag validation.

:class:`SystemPrompt` models the non-message system block from format spec
:class:`ToolSpec` describes each callable tool and its JSON-compatible
schema. The module reuses immutable containers from ``_base`` and JSON/tool
types from ``_primitives``. ``Conversation`` owns at most one system prompt,
and the future codec will render its fields in canonical tag order.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, ClassVar, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictStr,
    WithJsonSchema,
    field_validator,
)
from pydantic.config import JsonDict

from apertus_common.model._base import (
    FrozenList,
    ImmutableDict,
    ImmutableList,
    UnicodeScalarString,
)
from apertus_common.model._primitives import Effort, FiniteJsonObject, ToolName

TAG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
TAG_NAME_JSON_PATTERN = r"^(?!.*\s)[A-Za-z_][A-Za-z0-9_-]*$"
ExtraTagName = Annotated[
    StrictStr,
    WithJsonSchema({"type": "string", "pattern": TAG_NAME_JSON_PATTERN}),
]

CANONICAL_TAGS = frozenset(
    {
        "identity",
        "behavior",
        "effort",
        "tools",
        "tool",
        "schema",
        "policy",
        "environment",
    }
)


def _require_effort_string(value: object) -> object:
    if not isinstance(value, (str, Effort)):
        raise ValueError("effort must be a string or Effort value")  # noqa: TRY004
    return value


type EffortValue = Annotated[Effort, BeforeValidator(_require_effort_string)]


class ToolSpec(BaseModel):
    """Typed declaration for one system-prompt ``<tool>`` entry.

    ``name`` uses the same grammar as :class:`ToolCall`; ``json_schema`` must be
    recursively JSON-compatible and renders under ``<schema>``; optional
    ``policy`` tells the model when or how to use the tool.
    This layer does not validate that the object is a correct JSON Schema or
    that call arguments conform to it; those checks belong to the checker.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        validate_default=True,
    )

    name: ToolName
    json_schema: FiniteJsonObject
    policy: UnicodeScalarString | None = None


def _empty_tools() -> ImmutableList[ToolSpec]:
    return FrozenList(())


class SystemPrompt(BaseModel):
    """Typed contents of the non-message ``<|sys|> … <|/sys|>`` block.

    A :class:`Conversation` may own one prompt. It has authority rank 1 and no
    message header. ``effort`` is required; tool declarations
    and noncanonical tags remain immutable after validation. The future codec
    renders identity, behavior, effort, tools, extra tags, then environment
    last so volatile data does not invalidate the stable prefix unnecessarily.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        validate_default=True,
    )

    identity: UnicodeScalarString | None = None
    behavior: UnicodeScalarString | None = None
    effort: EffortValue
    tools: ImmutableList[ToolSpec] = Field(default_factory=_empty_tools)
    environment: UnicodeScalarString | None = None
    extra_tags: ImmutableDict[ExtraTagName, UnicodeScalarString] = Field(
        default_factory=dict,
        json_schema_extra=cast(
            JsonDict,
            {
                "propertyNames": {
                    "type": "string",
                    "pattern": TAG_NAME_JSON_PATTERN,
                    "not": {"enum": sorted(CANONICAL_TAGS)},
                }
            },
        ),
    )

    @field_validator("extra_tags")
    @classmethod
    def _check_extra_tags(cls, tags: Mapping[str, str]) -> Mapping[str, str]:
        for tag in tags:
            if TAG_NAME_RE.fullmatch(tag) is None:
                raise ValueError(f"tag name {tag!r} violates the tag-name grammar")
            if tag in CANONICAL_TAGS:
                raise ValueError(
                    f"tag name {tag!r} collides with a canonical system "
                    "prompt tag; use the typed field"
                )
        return tags
