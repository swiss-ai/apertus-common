"""Small fluent builders that delegate construction to the native models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self

from apertus_common.model._base import Message, Model
from apertus_common.model._conversation import Wait
from apertus_common.model._input import Document, Host, Result, User
from apertus_common.model._output import Call, Claim, Reply, Think
from apertus_common.model._primitives import Thinking
from apertus_common.model._system import SystemPrompt, ToolSpec

if TYPE_CHECKING:
    from apertus_common.checker import CheckMode, CheckOptions


class ModelBuilder[T: Model]:
    """Mutable construction state; each build returns an immutable snapshot."""

    def __init__(self, model: type[T]) -> None:
        self._model = model
        self._values: dict[str, Any] = {}

    def _set(self, field: str, value: object) -> Self:
        if field not in self._model.model_fields and not (
            field == "arguments" and issubclass(self._model, Call)
        ):
            raise ValueError(f"{self._model.__name__} has no builder field {field!r}")
        if field == "system" and field in self._values:
            raise ValueError("system is already assigned")
        self._values[field] = value
        return self

    def payload(self, value: str) -> Self:
        return self._set("payload", value)

    def loss_weight(self, value: float | None) -> Self:
        return self._set("loss_weight", value)

    def name(self, value: str) -> Self:
        return self._set("name", value)

    def counter(self, value: int) -> Self:
        return self._set("counter", value)

    def arguments(self, value: Mapping[str, Any]) -> Self:
        return self._set("arguments", value)

    def from_(self, value: str) -> Self:
        return self._set("from_", value)

    def id(self, value: str) -> Self:
        return self._set("id", value)

    def type(self, value: str) -> Self:
        return self._set("type", value)

    def output_type(self, value: str) -> Self:
        return self._set("output_type", value)

    def headers(self, value: Mapping[str, Any]) -> Self:
        return self._set("headers", value)

    def identity(self, value: str) -> Self:
        return self._set("identity", value)

    def thinking(self, value: Thinking | str) -> Self:
        return self._set("thinking", value)

    def behavior(self, value: str) -> Self:
        return self._set("behavior", value)

    def environment(self, value: str) -> Self:
        return self._set("environment", value)

    def description(self, value: str) -> Self:
        return self._set("description", value)

    def json_schema(self, value: Mapping[str, Any]) -> Self:
        return self._set("json_schema", value)

    def policy(self, value: str) -> Self:
        return self._set("policy", value)

    def tools(self, value: Sequence[ToolSpec]) -> Self:
        return self._set("tools", list(value))

    def tool(self, value: ToolSpec) -> Self:
        return self.tools([*self._values.get("tools", ()), value])

    def system(self, value: SystemPrompt) -> Self:
        return self._set("system", value)

    def items(self, value: Sequence[Message | Wait]) -> Self:
        return self._set("items", list(value))

    def message(self, value: Message | Wait) -> Self:
        return self.items([*self._values.get("items", ()), value])

    def host(self, payload: str, *, loss_weight: float | None = None) -> Self:
        return self.message(Host(payload=payload, loss_weight=loss_weight))

    def user(self, payload: str, *, loss_weight: float | None = None) -> Self:
        return self.message(User(payload=payload, loss_weight=loss_weight))

    def document(
        self,
        payload: str,
        *,
        from_: str | None = None,
        id: str | None = None,
        loss_weight: float | None = None,
    ) -> Self:
        return self.message(
            Document(payload=payload, from_=from_, id=id, loss_weight=loss_weight)
        )

    def result(
        self, *, name: str, counter: int, payload: str, loss_weight: float | None = None
    ) -> Self:
        return self.message(
            Result(name=name, counter=counter, payload=payload, loss_weight=loss_weight)
        )

    def call(
        self,
        *,
        name: str,
        counter: int,
        payload: str | None = None,
        arguments: Mapping[str, Any] | None = None,
        loss_weight: float | None = None,
    ) -> Self:
        return self.message(
            Call.build(
                name=name,
                counter=counter,
                payload=payload,
                arguments=arguments,
                loss_weight=loss_weight,
            )
        )

    def think(self, payload: str, *, loss_weight: float | None = None) -> Self:
        return self.message(Think(payload=payload, loss_weight=loss_weight))

    def reply(self, payload: str, *, loss_weight: float | None = None) -> Self:
        return self.message(Reply(payload=payload, loss_weight=loss_weight))

    def claim(self, payload: str, *, loss_weight: float | None = None) -> Self:
        return self.message(Claim(payload=payload, loss_weight=loss_weight))

    def wait(self, *, loss_weight: float | None = None) -> Self:
        return self.message(Wait(loss_weight=loss_weight))

    def build(self, *, check: CheckMode | CheckOptions | None = None) -> T:
        values = dict(self._values)
        if issubclass(self._model, SystemPrompt):
            values.setdefault("thinking", Thinking.MEDIUM)
        from apertus_common.model._base import apply_model_check

        if issubclass(self._model, Call) and "arguments" in values:
            return apply_model_check(self._model.build(**values), check)
        return apply_model_check(self._model.model_validate(values), check)
