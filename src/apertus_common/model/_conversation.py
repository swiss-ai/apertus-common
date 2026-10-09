"""Versioned native conversations preserve supplied item order."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import (
    Field,
    GetJsonSchemaHandler,
    TypeAdapter,
    field_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from apertus_common.model._base import (
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


MessageValue = Annotated[
    Host | User | Document | Result | Think | Reply | Claim | Call | UnknownOutput,
    Field(discriminator="type"),
]
ConversationItem = Annotated[
    Host
    | User
    | Document
    | Result
    | Think
    | Reply
    | Claim
    | Call
    | UnknownOutput
    | Wait,
    Field(discriminator="type"),
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
