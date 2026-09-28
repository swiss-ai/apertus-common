"""Reusable leaf types and validators for message and system fields.

This module defines effort levels, tool names and call IDs, recursively finite
JSON values, and tagged text payload parts. ``_system`` uses
the tool and JSON types; ``_input`` uses text payloads; and ``_output`` uses call IDs and arguments. These types validate and
serialize data only; the renderer owns their wire representation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    StrictBool,
    StrictInt,
    StrictStr,
    WithJsonSchema,
)

from apertus_common.errors import HeaderError
from apertus_common.model._base import (
    UNICODE_SCALAR_JSON_PATTERN,
    FrozenDict,
    FrozenList,
    ImmutableList,
    UnicodeScalarString,
    freeze_dict,
    freeze_list,
)


class Effort(StrEnum):
    """Requested reasoning effort for the required system tag.

    The format defines ``low``, ``medium``, and ``high``.
    :class:`SystemPrompt` accepts either these enum members or their string
    values and stores the normalized enum.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


TOOL_CALL_ID_RE = re.compile(r"^[A-Za-z0-9_:.\-]{1,64}$")
_CANONICAL_ID_RE = re.compile(r"^[A-Za-z0-9_]+:[0-9]+$")
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
TOOL_CALL_ID_JSON_PATTERN = r"^(?!.*\s)[A-Za-z0-9_:.\-]{1,64}$"
TOOL_NAME_JSON_PATTERN = r"^(?!.*\s)[A-Za-z0-9_]+$"


def _check_tool_call_id(value: str) -> str:
    if TOOL_CALL_ID_RE.fullmatch(value) is None:
        raise HeaderError(
            f"tool-call id {value!r} does not match the permissive id shape "
            "[A-Za-z0-9_:.-]{1,64}"
        )
    return value


ToolCallId = Annotated[
    StrictStr,
    WithJsonSchema(
        {
            "type": "string",
            "minLength": 1,
            "maxLength": 64,
            "pattern": TOOL_CALL_ID_JSON_PATTERN,
        }
    ),
    AfterValidator(_check_tool_call_id),
]
"""Tool-call id; permissive shape pending a spec ruling on the echoed
id. The model itself writes the stricter ``NAME:COUNTER`` convention;
see :func:`is_canonical_id`."""


def is_canonical_id(value: str) -> bool:
    """Return whether *value* follows the model-written ``NAME:COUNTER`` form.

    Tool-call fields accept a wider echoed-ID grammar for API compatibility;
    this predicate distinguishes the canonical form without constructing a
    model.
    """
    return _CANONICAL_ID_RE.fullmatch(value) is not None


def check_tool_name(value: str) -> str:
    """Validate and return a tool name using ``[A-Za-z0-9_]+``.

    ``ToolSpec`` and ``ToolCall`` share this validator so declarations, calls,
    and canonical call IDs use one name grammar.
    """
    if TOOL_NAME_RE.fullmatch(value) is None:
        raise ValueError(
            f"tool name {value!r} violates the tool-name grammar [A-Za-z0-9_]+"
        )
    return value


ToolName = Annotated[
    StrictStr,
    WithJsonSchema({"type": "string", "pattern": TOOL_NAME_JSON_PATTERN}),
    AfterValidator(check_tool_name),
]


def _require_json_list(value: object) -> object:
    if not isinstance(value, (list, FrozenList)):
        raise ValueError("JSON arrays must be lists")  # noqa: TRY004
    return cast(list[object] | FrozenList[object], value)


def _require_json_object(value: object) -> object:
    if not isinstance(value, (dict, FrozenDict)):
        raise ValueError("JSON objects must be dictionaries")  # noqa: TRY004
    return cast(dict[object, object] | FrozenDict[object, object], value)


def _require_json_float(value: object) -> object:
    if not isinstance(value, float):
        raise ValueError("JSON numbers must be int or float values")  # noqa: TRY004
    return value


type FiniteFloat = Annotated[
    float, BeforeValidator(_require_json_float), Field(allow_inf_nan=False)
]
UNICODE_SCALAR_SCHEMA = {
    "type": "string",
    "pattern": UNICODE_SCALAR_JSON_PATTERN,
}


def _finite_json_schema(anchor: str) -> dict[str, Any]:
    recursive_ref = {"$dynamicRef": f"#{anchor}"}
    return {
        "$dynamicAnchor": anchor,
        "anyOf": [
            {"type": "null"},
            {"type": "boolean"},
            {"type": "integer"},
            {"type": "number"},
            UNICODE_SCALAR_SCHEMA,
            {"type": "array", "items": recursive_ref},
            {
                "type": "object",
                "propertyNames": UNICODE_SCALAR_SCHEMA,
                "additionalProperties": recursive_ref,
            },
        ],
    }


FINITE_JSON_VALUE_SCHEMA = _finite_json_schema("finiteJsonValue")
FINITE_JSON_OBJECT_SCHEMA = {
    "type": "object",
    "propertyNames": UNICODE_SCALAR_SCHEMA,
    "additionalProperties": _finite_json_schema("finiteJsonObjectValue"),
}


def _serialize_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            cast(str, key): _serialize_json_value(item) for key, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sequence = cast(Sequence[object], value)
        return [_serialize_json_value(item) for item in sequence]
    return value


type FiniteJsonValue = Annotated[
    StrictBool
    | StrictInt
    | FiniteFloat
    | UnicodeScalarString
    | Annotated[
        Sequence[FiniteJsonValue],
        BeforeValidator(_require_json_list),
        AfterValidator(freeze_list),
    ]
    | Annotated[
        Mapping[UnicodeScalarString, FiniteJsonValue],
        BeforeValidator(_require_json_object),
        AfterValidator(freeze_dict),
    ]
    | None,
    WithJsonSchema(FINITE_JSON_VALUE_SCHEMA),
    PlainSerializer(_serialize_json_value, return_type=Any),
]
type FiniteJsonObject = Annotated[
    Mapping[UnicodeScalarString, FiniteJsonValue],
    BeforeValidator(_require_json_object),
    AfterValidator(freeze_dict),
    WithJsonSchema(FINITE_JSON_OBJECT_SCHEMA),
    PlainSerializer(_serialize_json_value, return_type=Any),
]


class TextPart(BaseModel):
    """A validated text run inside a message payload.

    A bare string passed as a ``User`` or ``Attachment`` payload is normalized
    to an immutable sequence containing one ``TextPart``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        json_schema_extra={"required": ["part"]},
    )

    part: Literal["text"] = "text"
    text: UnicodeScalarString


Part = Annotated[TextPart, Field(discriminator="part")]


def _coerce_payload(value: object) -> object:
    if isinstance(value, str):
        return [TextPart(text=value)]
    return value


Payload = Annotated[
    ImmutableList[Part],
    BeforeValidator(_coerce_payload, json_schema_input_type=str | list[Part]),
]
"""Immutable part sequence used by ``User`` and ``Attachment`` payloads.

A bare string is normalized to a single :class:`TextPart`; an explicit list may
contain multiple text parts.
"""
