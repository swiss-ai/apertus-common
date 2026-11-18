"""Immutable findings, including checks that lacked required evidence."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

from apertus_common.errors import ApertusFormatError

RULE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*/[a-z0-9]+(?:-[a-z0-9]+)*$")


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Location:
    scope: Literal["system", "items"] = "items"
    item: int | None = None
    field: str | None = None
    source: Literal["value", "prefix", "checkpoint"] = "value"

    def __post_init__(self) -> None:
        if self.scope not in ("system", "items"):
            raise ValueError("invalid location scope")
        if self.source not in ("value", "prefix", "checkpoint"):
            raise ValueError("invalid location source")
        if self.item is not None and (self.scope != "items" or self.item < 0):
            raise ValueError(
                "item indexes require items scope and a non-negative index"
            )

    def __str__(self) -> str:
        base = self.scope if self.item is None else f"items[{self.item}]"
        if self.source != "value":
            base = f"{self.source}.{base}"
        return base if self.field is None else f"{base}.{self.field}"


@dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    severity: Severity
    location: Location
    message: str

    def __post_init__(self) -> None:
        if RULE_ID_RE.fullmatch(self.rule) is None or not self.message:
            raise ValueError("a finding needs a family/rule ID and explanation")
        if not isinstance(cast(object, self.severity), Severity):
            raise TypeError("severity must be a Severity")


@dataclass(frozen=True, slots=True)
class Unevaluated:
    rule: str
    location: Location
    message: str

    def __post_init__(self) -> None:
        if RULE_ID_RE.fullmatch(self.rule) is None or not self.message:
            raise ValueError("an unevaluated check needs a rule ID and explanation")


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    violations: tuple[Violation, ...] = ()
    unevaluated: tuple[Unevaluated, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "violations", tuple(self.violations))
        object.__setattr__(self, "unevaluated", tuple(self.unevaluated))

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def complete(self) -> bool:
        return not self.unevaluated

    @property
    def errors(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity is Severity.WARNING)

    def __iter__(self) -> Iterator[Violation]:
        return iter(self.violations)

    def __len__(self) -> int:
        return len(self.violations)


class ConformanceError(ApertusFormatError):
    """Requested checks failed or could not all be evaluated."""

    def __init__(self, report: ConformanceReport) -> None:
        self.report = report
        super().__init__(
            f"{len(report.errors)} conformance error(s), "
            f"{len(report.unevaluated)} unevaluated check(s)"
        )
