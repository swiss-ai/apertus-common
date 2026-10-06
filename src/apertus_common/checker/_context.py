"""Caller-owned evidence for checking complete or retained histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from apertus_common.checker.rules._base import Checker
    from apertus_common.model import Conversation, SystemPrompt


type CheckMode = Literal["profile", "conversation", "full"]


@dataclass(frozen=True, slots=True)
class CallState:
    """A call observed before the retained history."""

    counter: int
    name: str
    status: Literal["pending", "result", "host-closed"] = "pending"

    def __post_init__(self) -> None:
        if type(self.counter) is not int or self.counter < 0:
            raise ValueError("checkpoint call counters must be non-negative integers")
        if type(self.name) is not str or self.status not in (
            "pending",
            "result",
            "host-closed",
        ):
            raise ValueError("invalid checkpoint call name or status")


@dataclass(frozen=True, slots=True)
class HistoryCheckpoint:
    """Known state before the prefix/value; None means unavailable evidence."""

    system: SystemPrompt | None = None
    next_call_counter: int | None = None
    calls: tuple[CallState, ...] | None = None
    document_ids: frozenset[str] | None = None
    waiting: bool | None = None

    def __post_init__(self) -> None:
        if self.next_call_counter is not None and (
            type(self.next_call_counter) is not int or self.next_call_counter < 0
        ):
            raise ValueError("next_call_counter must be a non-negative integer")
        if self.calls is not None:
            object.__setattr__(self, "calls", tuple(self.calls))
            counters = [call.counter for call in self.calls]
            if len(counters) != len(set(counters)):
                raise ValueError("checkpoint calls must have distinct counters")
            if self.next_call_counter is not None and any(
                counter >= self.next_call_counter for counter in counters
            ):
                raise ValueError(
                    "checkpoint call counters must precede the next counter"
                )
        if self.document_ids is not None:
            object.__setattr__(self, "document_ids", frozenset(self.document_ids))
        if self.waiting is not None and type(self.waiting) is not bool:
            raise ValueError("checkpoint waiting must be a boolean or None")


@dataclass(frozen=True, slots=True)
class HostClosure:
    """Explicit meaning of one host item; never inferred from its prose."""

    item: int
    counter: int
    name: str
    source: Literal["value", "prefix"] = "value"

    def __post_init__(self) -> None:
        if type(self.item) is not int or self.item < 0:
            raise ValueError("host item must be a non-negative integer")
        if type(self.counter) is not int:
            raise ValueError("host closure counter must be an integer")
        if type(self.name) is not str or self.source not in ("value", "prefix"):
            raise ValueError("invalid host closure name or source")


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Evidence and purpose, independent from the selected checker preset."""

    purpose: Literal["serving", "pretraining"] = "serving"
    history: Literal["complete", "partial"] = "complete"
    prefix: Conversation | None = None
    checkpoint: HistoryCheckpoint | None = None
    host_closures: tuple[HostClosure, ...] | None = None

    def __post_init__(self) -> None:
        if self.purpose not in ("serving", "pretraining"):
            raise ValueError("purpose must be serving or pretraining")
        if self.history not in ("complete", "partial"):
            raise ValueError("history must be complete or partial")
        if self.checkpoint is not None and self.history != "partial":
            raise ValueError("a checkpoint requires history='partial'")
        if self.host_closures is not None:
            object.__setattr__(self, "host_closures", tuple(self.host_closures))
            keys = [
                (closure.source, closure.item, closure.counter)
                for closure in self.host_closures
            ]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate host closure evidence")


@dataclass(frozen=True, slots=True)
class CheckOptions:
    """The same options accepted by model.check(), for checked construction."""

    mode: CheckMode = "full"
    checkers: tuple[Checker, ...] | None = None
    context: CheckContext | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("profile", "conversation", "full"):
            raise ValueError(f"unknown check mode: {self.mode!r}")
        if self.checkers is not None:
            object.__setattr__(self, "checkers", tuple(self.checkers))
