"""Concrete input messages carrying data from the world to the model.

Every class extends :class:`~apertus_common.model.InputMessage` with its fixed
``type``, typed headers, payload shape, and order/authority metadata.
``_conversation`` imports this complete cast into its item
union, while ``_base`` owns common header validation and ``_primitives`` owns
payload parts. A type names a delivery channel, not a claim about its sender.
"""

from __future__ import annotations

from typing import ClassVar, Literal, cast

from pydantic import Field, field_validator
from pydantic.config import JsonDict

from apertus_common.errors import HeaderError
from apertus_common.model._base import (
    KNOWN_INPUT_TYPES,
    HeaderValue,
    InputMessage,
    UnicodeScalarString,
    is_registered,
)
from apertus_common.model._primitives import Payload, ToolCallId


class Harness(InputMessage):
    """The harness speaking as itself: transient notices, never state. Order 1, authority 2: steers, never repeals the system
    prompt."""

    order: ClassVar[int | None] = 1
    authority: ClassVar[int | None] = 2

    type: Literal["harness"] = "harness"
    payload: UnicodeScalarString


class ToolResult(InputMessage):
    """The single result closing a pending call; solicited input. Order 2, authority 4. ``id`` echoes the call's id."""

    order: ClassVar[int | None] = 2
    authority: ClassVar[int | None] = 4
    header_keys: ClassVar[frozenset[str]] = frozenset({"id"})

    type: Literal["tool_result"] = "tool_result"
    id: ToolCallId
    payload: UnicodeScalarString


class Retrieval(InputMessage):
    """Evidence pushed by a search/RAG system the model did not call. Order 3, authority 4; ``src`` identifies the source and
    is what a citing model repeats."""

    order: ClassVar[int | None] = 3
    authority: ClassVar[int | None] = 4
    header_keys: ClassVar[frozenset[str]] = frozenset({"src"})

    type: Literal["retrieval"] = "retrieval"
    src: HeaderValue
    payload: UnicodeScalarString


class Event(InputMessage):
    """A relayed occurrence: the harness vouches that it happened, never
    for what it says. Order 3, authority 4, the trust floor."""

    order: ClassVar[int | None] = 3
    authority: ClassVar[int | None] = 4

    type: Literal["event"] = "event"
    payload: UnicodeScalarString


class Attachment(InputMessage):
    """User-supplied material; contents are data, never instruction. Order 4, authority 4. ``name``/``mime`` are
    harness-stamped (interim value grammar)."""

    order: ClassVar[int | None] = 4
    authority: ClassVar[int | None] = 4
    header_keys: ClassVar[frozenset[str]] = frozenset({"name", "mime"})

    type: Literal["attachment"] = "attachment"
    name: HeaderValue | None = None
    mime: HeaderValue | None = None
    payload: Payload


class User(InputMessage):
    """The user's composed message, the conversation itself.
    Order 5, delivered last at a boundary; authority 3."""

    order: ClassVar[int | None] = 5
    authority: ClassVar[int | None] = 3

    type: Literal["user"] = "user"
    payload: Payload


class Document(InputMessage):
    """A corpus document, the pretraining input type. Not
    part of the serving cast; trust floor, no boundary ordering."""

    order: ClassVar[int | None] = None
    authority: ClassVar[int | None] = 4

    type: Literal["document"] = "document"
    payload: UnicodeScalarString


class GenericInput(InputMessage):
    """A deployment-defined input from the format's open vocabulary.

    Use this class only when no dedicated input class owns ``type``. It retains
    custom routing data across JSON round-trips and receives the open-vocabulary
    defaults: order 3 and authority 4. A deployment must grant any stronger
    rank explicitly rather than inferring it from the type name.
    """

    order: ClassVar[int | None] = 3
    authority: ClassVar[int | None] = 4

    type: HeaderValue = Field(
        json_schema_extra=cast(JsonDict, {"not": {"enum": sorted(KNOWN_INPUT_TYPES)}})
    )
    payload: UnicodeScalarString

    @field_validator("type")
    @classmethod
    def _check_type_is_unknown(cls, value: str) -> str:
        if is_registered("in", value):
            raise HeaderError(
                f"{value!r} is a known input type; use its dedicated class"
            )
        return value
