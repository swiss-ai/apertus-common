"""Apertus 2 input messages."""

from typing import Literal

from pydantic import Field, StrictInt

from apertus_common.model._base import HeaderValue, InputMessage


class Host(InputMessage, frozen=True):
    type: Literal["host"] = "host"


class User(InputMessage, frozen=True):
    type: Literal["user"] = "user"


class Document(InputMessage, frozen=True):
    type: Literal["document"] = "document"
    from_: HeaderValue | None = Field(
        default=None, validation_alias="from", serialization_alias="from"
    )
    id: HeaderValue | None = None


class Result(InputMessage, frozen=True):
    type: Literal["result"] = "result"
    name: HeaderValue
    counter: StrictInt
