"""Explicit state and typed results for the OpenAI compatibility subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from apertus_common.errors import AdapterError


class OpenAIFunction(TypedDict):
    name: str
    arguments: str


class OpenAIToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: OpenAIFunction


class OpenAIAssistantMessage(TypedDict, total=False):
    role: Literal["assistant"]
    content: str | None
    reasoning_content: str
    tool_calls: list[OpenAIToolCall]


class OpenAIMessage(TypedDict, total=False):
    role: str
    content: str | None
    reasoning_content: str
    tool_calls: list[OpenAIToolCall]
    tool_call_id: str


class OpenAIToolFunction(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]


class OpenAITool(TypedDict):
    type: Literal["function"]
    function: OpenAIToolFunction


class OpenAIRequest(TypedDict, total=False):
    messages: list[OpenAIMessage]
    tools: list[OpenAITool]
    reasoning_effort: str


class OpenAIFunctionDelta(TypedDict, total=False):
    name: str
    arguments: str


class OpenAIToolDelta(TypedDict, total=False):
    index: int
    id: str
    type: Literal["function"]
    function: OpenAIFunctionDelta


class OpenAIDelta(TypedDict, total=False):
    role: Literal["assistant"]
    content: str
    reasoning_content: str
    tool_calls: list[OpenAIToolDelta]


class ToolAssociation(BaseModel):
    """Stable external ID associated with one native call."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    id: StrictStr = Field(min_length=1)
    name: StrictStr = Field(min_length=1)
    counter: StrictInt = Field(ge=0)


class OpenAIMappingContext(BaseModel):
    """Caller-owned, JSON-serializable associations surviving history compaction."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    associations: tuple[ToolAssociation, ...] = ()
    next_counter: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len({x.id for x in self.associations}) != len(self.associations):
            raise ValueError("duplicate tool IDs in mapping context")
        if len({x.counter for x in self.associations}) != len(self.associations):
            raise ValueError("duplicate counters in mapping context")
        if any(x.counter >= self.next_counter for x in self.associations):
            raise ValueError("next_counter must exceed every retained counter")
        return self


@dataclass(frozen=True, slots=True)
class MappingPolicy:
    merge_system_messages: bool = False
    merge_assistant_messages: bool = False
    omit_claims: bool = False
    omit_loss_weights: bool = False


@dataclass(frozen=True, slots=True)
class MappingDiagnostic:
    code: str
    index: int
    message: str
    lossy: bool = False


@dataclass(frozen=True, slots=True)
class MappingResult[T]:
    value: T
    diagnostics: tuple[MappingDiagnostic, ...]
    context: OpenAIMappingContext


def loss_weight_diagnostic(
    weight: float | None,
    policy: MappingPolicy,
    index: int,
    *,
    scope: Literal["message", "item", "system"] = "item",
) -> MappingDiagnostic | None:
    if weight is None:
        return None
    if not policy.omit_loss_weights:
        raise AdapterError(scope, index, "loss weights have no OpenAI counterpart")
    return MappingDiagnostic(
        "loss-weight-omitted", index, "Loss weight omitted from OpenAI mapping", True
    )


class Associations:
    """Mutable per-operation workspace; only immutable snapshots leave it."""

    def __init__(self, context: OpenAIMappingContext | None) -> None:
        context = context or OpenAIMappingContext()
        self.by_id = {item.id: item for item in context.associations}
        self.by_counter = {item.counter: item for item in context.associations}
        self.next_counter = context.next_counter

    def external(self, call_id: str, name: str, index: int) -> ToolAssociation:
        if not name:
            raise AdapterError("message", index, "function name must be nonempty")
        if call_id in self.by_id:
            association = self.by_id[call_id]
            if association.name != name:
                raise AdapterError(
                    "message", index, "tool ID is already associated with another name"
                )
            return association
        return self._add(call_id, name, self.next_counter)

    def native(
        self, counter: int, name: str, index: int, *, create: bool = True
    ) -> ToolAssociation:
        if counter in self.by_counter:
            association = self.by_counter[counter]
            if association.name != name:
                raise AdapterError(
                    "item", index, "counter is already associated with another name"
                )
            return association
        if not create or counter < 0:
            raise AdapterError("item", index, "unresolved result or invalid counter")
        call_id = f"apertus_{counter}"
        while call_id in self.by_id:
            call_id += "_"
        return self._add(call_id, name, counter)

    def _add(self, call_id: str, name: str, counter: int) -> ToolAssociation:
        association = ToolAssociation(id=call_id, name=name, counter=counter)
        self.by_id[call_id] = association
        self.by_counter[counter] = association
        self.next_counter = max(self.next_counter, counter + 1)
        return association

    def snapshot(self) -> OpenAIMappingContext:
        return OpenAIMappingContext(
            associations=tuple(self.by_counter.values()), next_counter=self.next_counter
        )
