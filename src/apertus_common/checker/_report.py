"""Violation and report types for conformance checking.

Plain frozen dataclasses, not pydantic models: reports are library outputs,
never validation boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

from apertus_common.errors import ApertusFormatError

Scope = Literal["system", "items", "stream"]
"""Where a violation lives: the system prompt, an item, or (reserved for
the optional later string/token check front ends) the raw stream between
messages."""

_RULE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*/[a-z0-9]+(?:-[a-z0-9]+)*$")


class Severity(StrEnum):
    """Weight of a conformance violation.

    ``ERROR`` breaks conformance (``ConformanceReport.ok`` is False);
    ``WARNING`` is advisory and never fails a report.
    """

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Location:
    """Position of a violation inside a conversation.

    ``item`` indexes ``Conversation.items`` (meaningful only for scope
    ``"items"``); ``field`` optionally names a path inside the located
    object, such as ``"args"`` or ``"tools[1].json_schema"``. Scope
    ``"stream"`` is reserved for the optional later check front ends over
    strings or token ids.
    """

    scope: Scope = "items"
    item: int | None = None
    field: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in ("system", "items", "stream"):
            raise ValueError(f"invalid location scope: {self.scope!r}")
        if self.item is not None:
            if self.scope != "items":
                raise ValueError("item can only be set for an items location")
            if self.item < 0:
                raise ValueError("item must be non-negative")
        if self.field is not None and not self.field:
            raise ValueError("field must be non-empty when set")

    def __str__(self) -> str:
        if self.scope == "items" and self.item is not None:
            base = f"items[{self.item}]"
        else:
            base = self.scope
        return base if self.field is None else f"{base}.{self.field}"


@dataclass(frozen=True, slots=True)
class Violation:
    """One typed conformance finding.

    ``rule`` is a stable ``family/rule`` id and part of the public contract;
    ``message`` is human-readable and not stable.
    """

    rule: str
    severity: Severity
    location: Location
    message: str

    def __post_init__(self) -> None:
        if _RULE_ID_RE.fullmatch(self.rule) is None:
            raise ValueError(
                f"rule id {self.rule!r} violates the family/rule kebab-case grammar"
            )
        if not isinstance(cast(object, self.severity), Severity):
            raise TypeError("severity must be a Severity")
        if not isinstance(cast(object, self.location), Location):
            raise TypeError("location must be a Location")
        if not self.message:
            raise ValueError("message must be non-empty")


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    """Immutable result of a conformance run.

    ``ok`` means no error-severity violations; warnings never fail a report.
    """

    violations: tuple[Violation, ...]

    def __post_init__(self) -> None:
        violations = cast(object, self.violations)
        if not isinstance(violations, tuple):
            raise TypeError("violations must be a tuple of Violation values")
        runtime_violations = cast(tuple[object, ...], violations)
        if not all(isinstance(value, Violation) for value in runtime_violations):
            raise TypeError("violations must be a tuple of Violation values")

    @property
    def ok(self) -> bool:
        """Whether the conversation is conformant (no errors)."""
        return not self.errors

    @property
    def errors(self) -> tuple[Violation, ...]:
        """The breaking violations."""
        return tuple(v for v in self.violations if v.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Violation, ...]:
        """The advisory violations."""
        return tuple(v for v in self.violations if v.severity is Severity.WARNING)

    def __iter__(self) -> Iterator[Violation]:
        return iter(self.violations)

    def __len__(self) -> int:
        return len(self.violations)


class ConformanceError(ApertusFormatError):
    """A conversation failed conformance where the caller asked to raise.

    Raised by the ``strict=True`` ingest flags (``from_json``,
    ``from_openai``) when the report has error-severity violations; warnings
    never raise. ``report`` carries the full ``ConformanceReport``.
    """

    def __init__(self, report: ConformanceReport) -> None:
        super().__init__(report)
        self.report = report

    def __str__(self) -> str:
        errors = self.report.errors
        shown = "; ".join(
            f"{v.rule} at {v.location.scope}"
            + (f" {v.location.item}" if v.location.item is not None else "")
            + f": {v.message}"
            for v in errors
        )
        return f"{len(errors)} conformance error(s): {shown}"
