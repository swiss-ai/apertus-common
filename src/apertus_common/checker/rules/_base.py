"""Rule declarations, the ``Checker`` base class, and burst iteration.

A checker implements one spec rule family. Subclass :class:`Checker`,
declare the rules it can emit, and pass an instance to ``check()`` to
extend conformance checking.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

from apertus_common.checker._report import Location, Severity, Violation
from apertus_common.model import Conversation, Wait

RULE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*/[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class Rule:
    """Declaration of one checkable rule.

    ``id`` is a stable ``family/rule`` kebab-case string and part of the
    public contract; the ``family/`` prefix is the namespace, and builtin
    family names are reserved (third parties pick a distinguishing
    family). ``description`` is one table-ready line and
    ``default_severity`` determines whether the rule initially produces
    breaking errors or advisory warnings.
    """

    id: str
    description: str
    default_severity: Severity

    def __post_init__(self) -> None:
        if RULE_ID_RE.fullmatch(self.id) is None:
            raise ValueError(
                f"rule id {self.id!r} violates the family/rule kebab-case grammar"
            )
        if not self.description.strip():
            raise ValueError(f"rule {self.id!r} has an empty description")
        if not isinstance(cast(object, self.default_severity), Severity):
            raise TypeError("default_severity must be a Severity")


class Checker(ABC):
    """One rule family of the conformance check.

    Subclasses declare ``rules`` and implement :meth:`check`. Severity
    resolution: each rule's declared default, then a whole-checker
    ``severity`` override, then per-rule ``overrides``. Instances are
    stateless and reusable across calls.

    Contract for implementations: never condition behavior on the
    resolved severity (call-level overrides remap severities after the
    run), and subclasses overriding ``__init__`` must call
    ``super().__init__()``.
    """

    rules: ClassVar[tuple[Rule, ...]] = ()

    def __init__(
        self,
        *,
        severity: Severity | None = None,
        overrides: Mapping[str, Severity] | None = None,
    ) -> None:
        ids = [rule.id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{type(self).__name__} declares duplicate rule ids")
        resolved = {rule.id: rule.default_severity for rule in self.rules}
        if severity is not None:
            resolved = dict.fromkeys(resolved, severity)
        if overrides:
            unknown = sorted(set(overrides) - set(resolved))
            if unknown:
                raise ValueError(f"{type(self).__name__} declares no rule(s) {unknown}")
            resolved.update(overrides)
        self._severities = resolved

    def violation(self, rule_id: str, location: Location, message: str) -> Violation:
        """Build a violation carrying this instance's resolved severity.

        A ``rule_id`` not declared in ``rules`` is a checker programming
        error and raises ``KeyError``.
        """
        return Violation(
            rule=rule_id,
            severity=self._severities[rule_id],
            location=location,
            message=message,
        )

    @abstractmethod
    def check(self, conversation: Conversation) -> Iterable[Violation]:
        """Yield every violation of this family found in *conversation*."""


def iter_bursts(conversation: Conversation) -> Iterator[range]:
    """Yield the index range of each maximal item run between waits.

    Ranges index ``Conversation.items`` directly so checkers report
    absolute locations. Empty runs between consecutive waits are skipped.
    """
    start = 0
    for index, item in enumerate(conversation.items):
        if isinstance(item, Wait):
            if index > start:
                yield range(start, index)
            start = index + 1
    if len(conversation.items) > start:
        yield range(start, len(conversation.items))
