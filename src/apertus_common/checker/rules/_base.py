"""Small extensible checker contract; checkers never mutate model values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from apertus_common.checker._context import CheckContext
from apertus_common.checker._report import (
    RULE_ID_RE,
    Location,
    Severity,
    Unevaluated,
    Violation,
)
from apertus_common.model import Conversation, Message, SystemPrompt, ToolSpec, Wait

type CheckValue = Conversation | Message | SystemPrompt | ToolSpec | Wait


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    description: str
    default_severity: Severity = Severity.ERROR

    def __post_init__(self) -> None:
        if RULE_ID_RE.fullmatch(self.id) is None or not self.description.strip():
            raise ValueError("rules need a family/rule ID and nonempty description")


class Checker(ABC):
    rules: ClassVar[tuple[Rule, ...]] = ()

    def violation(self, rule_id: str, location: Location, message: str) -> Violation:
        rule = next((rule for rule in self.rules if rule.id == rule_id), None)
        if rule is None:
            raise ValueError(f"checker does not declare {rule_id!r}")
        return Violation(rule_id, rule.default_severity, location, message)

    @abstractmethod
    def check(
        self, value: CheckValue, context: CheckContext | None = None
    ) -> Iterable[Violation | Unevaluated]:
        """Yield findings; absent evidence is Unevaluated, not a violation."""
