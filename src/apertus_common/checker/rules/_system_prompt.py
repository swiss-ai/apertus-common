"""The tool inventory of the system prompt.

The system prompt declares one tool entry per tool, with its JSON Schema
and usage policy. Tool names must resolve uniquely for a call's declared
name to mean anything, so duplicates break the inventory.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from apertus_common.checker._report import Location, Severity, Violation
from apertus_common.checker.rules._base import Checker, Rule
from apertus_common.model import Conversation


class SystemPromptChecker(Checker):
    """Declared tool names must be unique.

    Error by default: a duplicate name makes "the name of a declared tool" ambiguous. The typed model deliberately does not enforce
    this, so non-conformant prompts stay representable.
    """

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule(
            id="system/duplicate-tool-name",
            description="each declared tool name appears exactly once",
            default_severity=Severity.ERROR,
        ),
    )

    def check(self, conversation: Conversation) -> Iterable[Violation]:
        if conversation.system is None:
            return
        seen: set[str] = set()
        for index, spec in enumerate(conversation.system.tools):
            if spec.name in seen:
                yield self.violation(
                    "system/duplicate-tool-name",
                    Location(scope="system", field=f"tools[{index}]"),
                    f"tool name {spec.name!r} is declared more than once",
                )
            seen.add(spec.name)
