"""Finite immutable JSON and reasoning settings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, cast

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    PlainSerializer,
    StrictBool,
    StrictInt,
    WithJsonSchema,
)

from apertus_common.model._base import (
    HEADER_VALUE_JSON_PATTERN,
    UNICODE_SCALAR_JSON_PATTERN,
    FrozenDict,
    FrozenList,
    UnicodeScalarString,
    freeze_dict,
    freeze_list,
)


class Thinking(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _require_json_list(value: object) -> object:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("JSON arrays must be sequences")  # noqa: TRY004
    return cast(list[object] | FrozenList[object], value)


def _require_json_object(value: object) -> object:
    if not isinstance(value, Mapping):
        raise ValueError("JSON objects must be mappings")  # noqa: TRY004
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


def _finite_json_schema(
    anchor: str, string_schema: dict[str, str] = UNICODE_SCALAR_SCHEMA
) -> dict[str, Any]:
    recursive_ref = {"$dynamicRef": f"#{anchor}"}
    return {
        "$dynamicAnchor": anchor,
        "anyOf": [
            {"type": "null"},
            {"type": "boolean"},
            {"type": "integer"},
            {"type": "number"},
            string_schema,
            {"type": "array", "items": recursive_ref},
            {
                "type": "object",
                "propertyNames": string_schema,
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


type JsonObject = FiniteJsonObject

_HEADER_STRING_SCHEMA = {"type": "string", "pattern": HEADER_VALUE_JSON_PATTERN}
type HeaderJsonObject = Annotated[
    FiniteJsonObject,
    WithJsonSchema(
        {
            "type": "object",
            "propertyNames": _HEADER_STRING_SCHEMA,
            "additionalProperties": _finite_json_schema(
                "headerJsonValue", _HEADER_STRING_SCHEMA
            ),
        }
    ),
]
