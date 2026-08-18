"""Foundational validation and metadata shared by the model package.

This module owns immutable sequence/mapping storage, strict Unicode and header
types, the :class:`TypeInfo` registry, and the input/output message base
classes. ``_primitives`` builds reusable field types on these foundations;
``_input`` and ``_output`` define the concrete cast; and ``_conversation`` uses
the registry for discrimination and canonical ordering. Order and authority
are class metadata, never serialized fields.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Never,
    TypeVar,
    cast,
    overload,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    PlainSerializer,
    StrictStr,
    WithJsonSchema,
    field_validator,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue

from apertus_common.errors import HeaderError

Direction = Literal["in", "out"]

HEADER_KEY_RE = re.compile(r"^[A-Za-z0-9_]+$")
HEADER_KEY_JSON_PATTERN = r"^(?!.*\s)[A-Za-z0-9_]+$"
HEADER_VALUE_JSON_PATTERN = (
    r"^(?!.*[\uD800-\uDFFF])(?!.*\s)(?!.*=)(?!.*<\|)(?!.*\|>)[\s\S]+$"
)
UNICODE_SCALAR_JSON_PATTERN = r"^(?!.*[\uD800-\uDFFF])[\s\S]*$"

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")


def _immutable() -> Never:
    raise TypeError("validated model containers are immutable")


class FrozenList(Sequence[_T]):
    """Sequence implementation that cannot change after construction.

    Pydantic's ``ImmutableList`` annotation converts validated list-like fields
    to this storage type. It behaves as a :class:`collections.abc.Sequence`
    inside models and serializes back to an ordinary JSON array.
    """

    __slots__ = ("__items",)
    __items: tuple[_T, ...]

    def __init__(self, values: Iterable[_T]) -> None:
        if hasattr(self, "_FrozenList__items"):
            _immutable()
        object.__setattr__(self, "_FrozenList__items", tuple(values))

    def __setattr__(self, name: str, value: object) -> Never:
        _immutable()

    @overload
    def __getitem__(self, key: int) -> _T: ...

    @overload
    def __getitem__(self, key: slice) -> FrozenList[_T]: ...

    def __getitem__(self, key: int | slice) -> _T | FrozenList[_T]:
        if isinstance(key, slice):
            return FrozenList(self.__items[key])
        return self.__items[key]

    def __iter__(self) -> Iterator[_T]:
        return iter(self.__items)

    def __len__(self) -> int:
        return len(self.__items)

    def __repr__(self) -> str:
        return repr(list(self.__items))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return tuple(self) == tuple(cast(Sequence[object], other))
        return False

    def __copy__(self) -> FrozenList[_T]:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenList[_T]:
        return self


class FrozenDict(Mapping[_K, _V]):
    """Mapping implementation that cannot change after construction.

    Pydantic's ``ImmutableDict`` annotation converts validated mapping fields
    to this storage type. Its private proxy prevents mutation while serializers
    expose an ordinary JSON object.
    """

    __slots__ = ("__items",)
    __items: Mapping[_K, _V]

    def __init__(self, values: Mapping[_K, _V]) -> None:
        if hasattr(self, "_FrozenDict__items"):
            _immutable()
        object.__setattr__(self, "_FrozenDict__items", MappingProxyType(dict(values)))

    def __setattr__(self, name: str, value: object) -> Never:
        _immutable()

    def __getitem__(self, key: _K) -> _V:
        return self.__items[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(self.__items)

    def __len__(self) -> int:
        return len(self.__items)

    def __repr__(self) -> str:
        return repr(dict(self.__items))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self) == dict(
            cast(Mapping[object, object], other)
        )

    def __copy__(self) -> FrozenDict[_K, _V]:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenDict[_K, _V]:
        return self


def freeze_list[T](value: Sequence[T]) -> FrozenList[T]:
    """Convert a validated sequence to the model package's immutable storage.

    Existing :class:`FrozenList` values are returned unchanged so nested model
    revalidation remains cheap and idempotent.
    """
    return value if isinstance(value, FrozenList) else FrozenList(value)


def freeze_dict[K, V](value: Mapping[K, V]) -> FrozenDict[K, V]:
    """Convert a validated mapping to the model package's immutable storage.

    Existing :class:`FrozenDict` values are returned unchanged so nested model
    revalidation remains cheap and idempotent.
    """
    return value if isinstance(value, FrozenDict) else FrozenDict(value)


def _serialize_list[T](value: Sequence[T]) -> list[T]:
    return list(value)


def _serialize_dict[K, V](value: Mapping[K, V]) -> dict[K, V]:
    return dict(value)


type ImmutableList[T] = Annotated[
    Sequence[T],
    AfterValidator(freeze_list),
    PlainSerializer(_serialize_list, return_type=list[T]),
]
type ImmutableDict[K, V] = Annotated[
    Mapping[K, V],
    AfterValidator(freeze_dict),
    PlainSerializer(_serialize_dict, return_type=dict[K, V]),
]


def check_unicode_scalar(value: str) -> str:
    """Return *value* after ensuring it can make a lossless JSON round-trip.

    Lone UTF-16 surrogates are not Unicode scalar values and cannot be encoded
    by Pydantic's JSON serializer, so all model strings reject them eagerly.
    """
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("strings must contain Unicode scalar values only")
    return value


def check_header_value(value: str) -> str:
    """Validate and return a value safe for the current header grammar.

    Values must be non-empty and contain no whitespace, ``=``, or control-glyph
    fragments (``<|`` and ``|>``). Unsafe values are rejected rather than
    escaped while the format specification's escaping decision remains open.
    """
    if not value:
        raise HeaderError("header value must be non-empty")
    if any(ch.isspace() for ch in value):
        raise HeaderError(f"header value {value!r} contains whitespace")
    if "=" in value:
        raise HeaderError(f"header value {value!r} contains '='")
    if "<|" in value or "|>" in value:
        raise HeaderError(f"header value {value!r} contains a control-glyph sequence")
    return value


HeaderKey = Annotated[
    StrictStr,
    WithJsonSchema({"type": "string", "pattern": HEADER_KEY_JSON_PATTERN}),
]
UnicodeScalarString = Annotated[
    StrictStr,
    WithJsonSchema({"type": "string", "pattern": UNICODE_SCALAR_JSON_PATTERN}),
    AfterValidator(check_unicode_scalar),
]
HeaderValue = Annotated[
    StrictStr,
    WithJsonSchema(
        {"type": "string", "minLength": 1, "pattern": HEADER_VALUE_JSON_PATTERN}
    ),
    AfterValidator(check_unicode_scalar),
    AfterValidator(check_header_value),
]


@dataclass(frozen=True, slots=True)
class TypeInfo:
    """Order and authority labels for one message type.

    ``order``: canonical delivery position at a boundary (1 harness …
    5 user). ``authority``: trust rank (1 system prompt, 2 harness,
    3 user, 4 data floor). ``None`` where the spec assigns none
    (output types; the pretraining ``document``).
    """

    type: str
    direction: Direction
    order: int | None
    authority: int | None


_TYPE_INFOS = (
    TypeInfo("harness", "in", 1, 2),
    TypeInfo("tool_result", "in", 2, 4),
    TypeInfo("retrieval", "in", 3, 4),
    TypeInfo("event", "in", 3, 4),
    TypeInfo("attachment", "in", 4, 4),
    TypeInfo("user", "in", 5, 3),
    TypeInfo("document", "in", None, 4),
    TypeInfo("assistant", "out", None, None),
    TypeInfo("think", "out", None, None),
    TypeInfo("tool_call", "out", None, None),
    TypeInfo("verifiable_answer", "out", None, None),
)
_REGISTRY: dict[tuple[str, str], TypeInfo] = {
    (info.direction, info.type): info for info in _TYPE_INFOS
}
KNOWN_INPUT_TYPES = frozenset(
    info.type for info in _TYPE_INFOS if info.direction == "in"
)
KNOWN_OUTPUT_TYPES = frozenset(
    info.type for info in _TYPE_INFOS if info.direction == "out"
)


def type_info(direction: Direction, type: str) -> TypeInfo:
    """Return order and authority metadata for a message type.

    Registered built-in types use the format table. An unregistered input type
    receives the open-vocabulary defaults (order 3, authority 4); unregistered
    outputs have neither label. ``Conversation.normalized()`` uses this
    function as its single ordering source.
    """
    if direction not in ("in", "out"):
        raise ValueError(f"invalid message direction: {direction!r}")
    info = _REGISTRY.get((direction, type))
    if info is not None:
        return info
    if direction == "in":
        return TypeInfo(type=type, direction="in", order=3, authority=4)
    return TypeInfo(type=type, direction="out", order=None, authority=None)


def is_registered(direction: str, type: str) -> bool:
    """True if (*direction*, *type*) names a dedicated message class.

    Package-internal helper for sibling modules; not part of the public API.
    """
    return (direction, type) in _REGISTRY


class _MessageBase(BaseModel):
    """Internal Pydantic base for the common message header shape.

    It freezes instances, rejects undeclared model fields, validates arbitrary
    ``extras`` against the open header grammar, and keeps typed header names out
    of ``extras``. :class:`InputMessage` and
    :class:`OutputMessage` add the serialized direction; concrete cast classes
    add ``type``, payload, and typed headers.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
    )

    order: ClassVar[int | None] = None
    authority: ClassVar[int | None] = None
    header_keys: ClassVar[frozenset[str]] = frozenset()

    extras: ImmutableDict[HeaderKey, HeaderValue] = Field(default_factory=dict)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: object, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Force defaulted discriminators into ``required`` and expose the
        reserved-header exclusions that validators enforce at runtime."""
        schema = cast(dict[str, object], handler(cast(Any, core_schema)))
        properties_value = schema.get("properties")
        if isinstance(properties_value, dict):
            properties = cast(dict[str, object], properties_value)
            required_value = schema.setdefault("required", [])
            if isinstance(required_value, list):
                required = cast(list[object], required_value)
                for discriminator in ("direction", "type"):
                    if discriminator in properties and discriminator not in required:
                        required.append(discriminator)
            extras_schema_value = properties.get("extras")
        else:
            extras_schema_value = None
        if isinstance(extras_schema_value, dict):
            extras_schema = cast(dict[str, object], extras_schema_value)
            property_names_value = extras_schema.setdefault("propertyNames", {})
            if isinstance(property_names_value, dict):
                property_names = cast(dict[str, object], property_names_value)
                property_names["type"] = "string"
                property_names["pattern"] = HEADER_KEY_JSON_PATTERN
                property_names["not"] = {
                    "enum": sorted({"type", "direction", *cls.header_keys})
                }
        return cast(JsonSchemaValue, schema)

    @field_validator("extras")
    @classmethod
    def _check_extras(cls, extras: Mapping[str, str]) -> Mapping[str, str]:
        # Values are validated by the HeaderValue annotation; only key
        # grammar and reserved-key collisions are checked here.
        for key in extras:
            if HEADER_KEY_RE.fullmatch(key) is None:
                raise HeaderError(
                    f"header key {key!r} violates the key grammar [A-Za-z0-9_]+"
                )
            if key in ("type", "direction") or key in cls.header_keys:
                raise HeaderError(
                    f"header key {key!r} duplicates a typed field of {cls.__name__}"
                )
        return extras


class InputMessage(_MessageBase):
    """Base for world-to-model messages framed as ``<|in|> … <|/in|>``.

    Concrete input classes add a fixed or deployment-defined ``type`` plus
    their payload and headers. Use those concrete classes for
    construction; this base is primarily useful for shared typing.
    """

    direction: Literal["in"] = "in"

    @model_validator(mode="before")
    @classmethod
    def _reject_direct_construction(cls, value: Any) -> Any:
        if cls is InputMessage:
            raise ValueError(
                "InputMessage is a typing base; use a concrete input class"
            )
        return value


class OutputMessage(_MessageBase):
    """Base for model-to-world messages framed as ``<|out|> … <|/out|>``.

    Concrete output classes add a fixed or deployment-defined routing ``type``
    plus their payload and headers. Use those concrete classes
    for construction; this base is primarily useful for shared typing.
    """

    direction: Literal["out"] = "out"

    @model_validator(mode="before")
    @classmethod
    def _reject_direct_construction(cls, value: Any) -> Any:
        if cls is OutputMessage:
            raise ValueError(
                "OutputMessage is a typing base; use a concrete output class"
            )
        return value
