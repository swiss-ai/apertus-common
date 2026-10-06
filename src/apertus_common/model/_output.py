"""Apertus 2 outputs preserve payloads independently of conformance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import Field, StrictInt, TypeAdapter, field_validator

from apertus_common.model._base import (
    KNOWN_OUTPUT_TYPES,
    HeaderValue,
    MessageType,
    OutputMessage,
    check_header_value,
)
from apertus_common.model._primitives import FiniteJsonObject, HeaderJsonObject

if TYPE_CHECKING:
    from apertus_common.checker import CheckMode, CheckOptions


class Think(OutputMessage, frozen=True):
    type: Literal["think"] = "think"


class Reply(OutputMessage, frozen=True):
    type: Literal["reply"] = "reply"


class Claim(OutputMessage, frozen=True):
    type: Literal["claim"] = "claim"


class Call(OutputMessage, frozen=True):
    type: Literal["call"] = "call"
    name: HeaderValue
    counter: StrictInt

    @classmethod
    def build(
        cls,
        *,
        name: str,
        counter: int,
        payload: str | None = None,
        arguments: Mapping[str, Any] | None = None,
        check: CheckMode | CheckOptions | None = None,
    ) -> Self:
        if (payload is None) == (arguments is None):
            raise ValueError("provide exactly one of payload and arguments")
        if arguments is not None:
            adapter: TypeAdapter[FiniteJsonObject] = TypeAdapter(FiniteJsonObject)
            payload = adapter.dump_json(adapter.validate_python(arguments)).decode()
        assert payload is not None
        from apertus_common.model._base import apply_model_check

        return apply_model_check(
            cls(name=name, counter=counter, payload=payload), check
        )


def validate_header_json(value: object) -> None:
    """Enforce the no-decoded-whitespace rule recursively."""
    if isinstance(value, str):
        check_header_value(value)
    elif isinstance(value, Mapping):
        for key, item in cast(Mapping[str, object], value).items():
            check_header_value(key)
            validate_header_json(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in cast(Sequence[object], value):
            validate_header_json(item)


class UnknownOutput(OutputMessage, frozen=True):
    type: MessageType = Field(
        json_schema_extra={
            "not": {"enum": [name for name in sorted(KNOWN_OUTPUT_TYPES)]}
        }
    )
    headers: HeaderJsonObject = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _unknown(cls, value: str) -> str:
        if value in KNOWN_OUTPUT_TYPES:
            raise ValueError("known output type requires its dedicated model")
        return value

    @field_validator("headers")
    @classmethod
    def _headers(cls, value: FiniteJsonObject) -> FiniteJsonObject:
        validate_header_json(value)
        return value
