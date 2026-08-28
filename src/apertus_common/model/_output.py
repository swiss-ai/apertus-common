"""Concrete output messages carrying model actions back to the world.

Every class extends :class:`~apertus_common.model.OutputMessage` with a fixed
or deployment-defined routing ``type`` and its payload fields (format spec
``_conversation`` imports the cast for item discrimination; future codecs
render these typed values, while the checker validates cross-message behavior
such as tool schema conformance. Output types carry no order or authority.
"""

from __future__ import annotations

from typing import ClassVar, Literal, cast

from pydantic import Field, field_validator
from pydantic.config import JsonDict

from apertus_common.errors import HeaderError
from apertus_common.model._base import (
    KNOWN_OUTPUT_TYPES,
    HeaderValue,
    OutputMessage,
    UnicodeScalarString,
    is_registered,
)
from apertus_common.model._primitives import FiniteJsonValue, ToolCallId, ToolName


class Assistant(OutputMessage):
    """The reply shown to the user; may carry soft structure the
    interface renders. Never stripped."""

    type: Literal["assistant"] = "assistant"
    payload: UnicodeScalarString


class Think(OutputMessage):
    """Reasoning stripped by memory policy; durable conclusions belong elsewhere."""

    type: Literal["think"] = "think"
    payload: UnicodeScalarString


class ToolCall(OutputMessage):
    """One call to one tool. The fixed wire grammar
    ``{"name": NAME, "args": ARGS}`` is held as typed fields and
    produced/parsed by the codec layer; ``args``-vs-schema conformance
    is a checker concern. ``id`` follows ``NAME:COUNTER`` when model-written."""

    header_keys: ClassVar[frozenset[str]] = frozenset({"id"})

    type: Literal["tool_call"] = "tool_call"
    id: ToolCallId
    name: ToolName
    args: FiniteJsonValue


class VerifiableAnswer(OutputMessage):
    """The task's answer as a byte-exact payload for verifiers/RLVR; shape is verifier-defined. A committed claim, never
    stripped."""

    type: Literal["verifiable_answer"] = "verifiable_answer"
    payload: UnicodeScalarString


class GenericOutput(OutputMessage):
    """A deployment-defined output from the format's open vocabulary.

    Use this class only when no dedicated output class owns ``type`` (format
    It preserves the payload for deployment-specific routing;
    deciding whether an unknown or merely unconsumed type needs a harness
    notice is a serving/checker concern, not model validation.
    """

    type: HeaderValue = Field(
        json_schema_extra=cast(JsonDict, {"not": {"enum": sorted(KNOWN_OUTPUT_TYPES)}})
    )
    payload: UnicodeScalarString

    @field_validator("type")
    @classmethod
    def _check_type_is_unknown(cls, value: str) -> str:
        if is_registered("out", value):
            raise HeaderError(
                f"{value!r} is a known output type; use its dedicated class"
            )
        return value
