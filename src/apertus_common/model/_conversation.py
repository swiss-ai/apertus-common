"""Versioned native conversations preserve supplied item order."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, cast

from pydantic import (
    Discriminator,
    Field,
    GetJsonSchemaHandler,
    Tag,
    TypeAdapter,
    field_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from apertus_common.model._base import (
    KNOWN_INPUT_TYPES,
    KNOWN_OUTPUT_TYPES,
    ImmutableList,
    LossWeight,
    Model,
)
from apertus_common.model._input import Document, Host, Result, User
from apertus_common.model._output import Call, Claim, Reply, Think, UnknownOutput
from apertus_common.model._system import SystemPrompt

if TYPE_CHECKING:
    from apertus_common.checker import CheckMode


class Wait(Model, frozen=True):
    type: Literal["wait"] = "wait"
    loss_weight: LossWeight = None


def _item_tag(value: Any) -> str | None:
    direction = (
        cast(Mapping[str, Any], value).get("direction")
        if isinstance(value, Mapping)
        else getattr(value, "direction", None)
    )
    kind = (
        cast(Mapping[str, Any], value).get("type")
        if isinstance(value, Mapping)
        else getattr(value, "type", None)
    )
    if direction is None and kind == "wait":
        return "wait"
    if not isinstance(kind, str):
        return None
    if direction == "in" and kind in KNOWN_INPUT_TYPES:
        return f"in:{kind}"
    if direction == "out":
        return f"out:{kind}" if kind in KNOWN_OUTPUT_TYPES else "out:*"
    return None


MessageValue = Annotated[
    Annotated[Host, Tag("in:host")]
    | Annotated[User, Tag("in:user")]
    | Annotated[Document, Tag("in:document")]
    | Annotated[Result, Tag("in:result")]
    | Annotated[Think, Tag("out:think")]
    | Annotated[Reply, Tag("out:reply")]
    | Annotated[Claim, Tag("out:claim")]
    | Annotated[Call, Tag("out:call")]
    | Annotated[UnknownOutput, Tag("out:*")],
    Discriminator(_item_tag),
]
ConversationItem = Annotated[
    Annotated[Host, Tag("in:host")]
    | Annotated[User, Tag("in:user")]
    | Annotated[Document, Tag("in:document")]
    | Annotated[Result, Tag("in:result")]
    | Annotated[Think, Tag("out:think")]
    | Annotated[Reply, Tag("out:reply")]
    | Annotated[Claim, Tag("out:claim")]
    | Annotated[Call, Tag("out:call")]
    | Annotated[UnknownOutput, Tag("out:*")]
    | Annotated[Wait, Tag("wait")],
    Discriminator(_item_tag),
]
MESSAGE_ADAPTER: TypeAdapter[MessageValue] = TypeAdapter(MessageValue)


class Conversation(Model, frozen=True):
    _default_check_mode: ClassVar[CheckMode] = "full"
    schema_version: Literal[2] = 2
    system: SystemPrompt | None = None
    items: ImmutableList[ConversationItem] = Field(default_factory=tuple)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        schema["$id"] = "urn:apertus-common:native:2"
        return schema

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value
