"""Apertus 2 input messages."""

from typing import Literal

from pydantic import Field, StrictInt

from apertus_common.model._base import InputMessage, NonEmptyHeaderValue


class Host(InputMessage, frozen=True):
    type: Literal["host"] = "host"


class User(InputMessage, frozen=True):
    type: Literal["user"] = "user"


class Document(InputMessage, frozen=True):
    type: Literal["document"] = "document"
    from_: NonEmptyHeaderValue | None = Field(
        default=None, validation_alias="from", serialization_alias="from"
    )
    id: NonEmptyHeaderValue | None = None


class Result(InputMessage, frozen=True):
    type: Literal["result"] = "result"
    name: NonEmptyHeaderValue
    counter: StrictInt
