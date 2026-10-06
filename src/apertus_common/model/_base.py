"""Immutable model foundations and strict native JSON loading."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    Never,
    Self,
    TypeVar,
    cast,
    overload,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    GetJsonSchemaHandler,
    PlainSerializer,
    StrictStr,
    ValidationError,
    ValidationInfo,
    WithJsonSchema,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

if TYPE_CHECKING:
    from apertus_common.checker import (
        CheckContext,
        Checker,
        CheckMode,
        CheckOptions,
        ConformanceReport,
    )
    from apertus_common.checker.rules._base import CheckValue
    from apertus_common.model._builders import ModelBuilder

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")
UNICODE_SCALAR_JSON_PATTERN = r"^[^\uD800-\uDFFF]*$"
HEADER_VALUE_JSON_PATTERN = r"^[^\s\uD800-\uDFFF]*(?![\s\S])"
MESSAGE_TYPE_JSON_PATTERN = r"^[A-Za-z0-9_-]+(?![\s\S])"


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

    def __delattr__(self, name: str) -> Never:
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

    def __delattr__(self, name: str) -> Never:
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
    if any(character.isspace() for character in value):
        raise ValueError("decoded header strings must not contain whitespace")
    return check_unicode_scalar(value)


UnicodeScalarString = Annotated[
    StrictStr,
    AfterValidator(check_unicode_scalar),
    WithJsonSchema({"type": "string", "pattern": UNICODE_SCALAR_JSON_PATTERN}),
]
HeaderValue = Annotated[
    UnicodeScalarString,
    AfterValidator(check_header_value),
    WithJsonSchema({"type": "string", "pattern": HEADER_VALUE_JSON_PATTERN}),
]


def check_type(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError(
            "message type must be a nonempty word containing letters, digits, underscore or hyphen"
        )
    return value


MessageType = Annotated[
    UnicodeScalarString,
    AfterValidator(check_type),
    WithJsonSchema({"type": "string", "pattern": MESSAGE_TYPE_JSON_PATTERN}),
]


def strict_json_loads(data: str | bytes) -> Any:
    """Decode strict JSON, rejecting duplicate keys and non-finite numbers."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> Never:
        raise ValueError(f"non-finite JSON number: {value}")

    def number(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result

    return json.loads(
        data, object_pairs_hook=pairs, parse_constant=constant, parse_float=number
    )


def json_validation_error(
    cls: type[BaseModel], value: object, error: ValueError
) -> ValidationError:
    return ValidationError.from_exception_data(
        cls.__name__,
        [{"type": "value_error", "loc": (), "input": value, "ctx": {"error": error}}],
    )


class Model(
    BaseModel,
    frozen=True,
    extra="forbid",
    allow_inf_nan=False,
    validate_default=True,
    revalidate_instances="never",
    populate_by_name=True,
):
    """Frozen native data with explicit optional conformance checking."""

    _default_check_mode: ClassVar[CheckMode] = "profile"

    @model_validator(mode="before")
    @classmethod
    def _native_aliases(cls, value: Any, info: ValidationInfo) -> Any:
        # Alias flags alone do not reach models inside Pydantic Sequence validators.
        if info.context == "native_json" and isinstance(value, Mapping):
            for name, field in cls.model_fields.items():
                alias = field.validation_alias
                if isinstance(alias, str) and alias != name and name in value:
                    raise ValueError(f"native JSON uses {alias!r}, not {name!r}")
        return cast(Any, value)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        properties = schema.get("properties", {})
        required = schema.setdefault("required", [])
        for field in ("direction", "type", "schema_version"):
            if field in properties and field not in required:
                required.append(field)
        return schema

    @classmethod
    def from_json(
        cls, data: str | bytes, *, check: CheckMode | CheckOptions | None = None
    ) -> Self:
        try:
            value = strict_json_loads(data)
            _require_discriminators(cls, value)
        except (ValueError, UnicodeError) as error:
            raise json_validation_error(cls, data, error) from error
        if cls in (Message, InputMessage, OutputMessage):
            from apertus_common.model._conversation import MESSAGE_ADAPTER

            dispatched = MESSAGE_ADAPTER.validate_python(
                value, by_alias=True, by_name=False, context="native_json"
            )
            if (cls is InputMessage and dispatched.direction != "in") or (
                cls is OutputMessage and dispatched.direction != "out"
            ):
                raise json_validation_error(
                    cls, value, ValueError("incorrect message direction")
                )
            return apply_model_check(cast(Self, dispatched), check)
        return apply_model_check(
            cls.model_validate(
                value, by_alias=True, by_name=False, context="native_json"
            ),
            check,
        )

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)

    def check(
        self,
        mode: CheckMode | None = None,
        *,
        checkers: Sequence[Checker] | None = None,
        context: CheckContext | None = None,
    ) -> ConformanceReport:
        from apertus_common.checker import check_value

        return check_value(
            cast("CheckValue", self),
            mode or self._default_check_mode,
            checkers=checkers,
            context=context,
        )

    @classmethod
    def builder(cls) -> ModelBuilder[Self]:
        from apertus_common.model._builders import ModelBuilder

        return ModelBuilder(cls)


def apply_model_check[T: Model](value: T, option: CheckMode | CheckOptions | None) -> T:
    if option is not None:
        from apertus_common.checker import apply_check

        apply_check(cast("CheckValue", value), option)
    return value


def _require_discriminators(cls: type[Model], value: object) -> None:
    if not isinstance(value, dict):
        return
    data = cast(dict[str, object], value)
    if "schema_version" in cls.model_fields:
        if "schema_version" not in data:
            raise ValueError("conversation JSON requires schema_version")
        items = data.get("items", [])
        if isinstance(items, list):
            for item in cast(list[object], items):
                _require_message_discriminators(item)
    elif "type" in cls.model_fields:
        _require_message_discriminators(data)


def _require_message_discriminators(value: object) -> None:
    if not isinstance(value, dict):
        return
    data = cast(dict[str, object], value)
    if "type" not in data or (data.get("type") != "wait" and "direction" not in data):
        raise ValueError("native messages require direction and type discriminators")


class Message(Model, frozen=True):
    """Common base for native messages; construct a concrete message type."""

    direction: Literal["in", "out"]
    type: MessageType
    payload: UnicodeScalarString

    @model_validator(mode="before")
    @classmethod
    def _concrete(cls, value: Any) -> Any:
        if cls in (Message, InputMessage, OutputMessage):
            raise ValueError("construct a concrete message type")
        return value


class InputMessage(Message, frozen=True):
    direction: Literal["in"] = "in"


class OutputMessage(Message, frozen=True):
    direction: Literal["out"] = "out"


KNOWN_INPUT_TYPES = frozenset({"host", "result", "user", "document"})
KNOWN_OUTPUT_TYPES = frozenset({"think", "reply", "claim", "call"})
