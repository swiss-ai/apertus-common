"""Verifiable answer placement within a burst.

Per the format spec, a verifiable answer comes after the think messages
and tool calls it rests on and canonically before the assistant message;
a task's parts go into a single verifiable answer carrying all of them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from apertus_common.checker._report import Location, Severity, Violation
from apertus_common.checker.rules._base import Checker, Rule, iter_bursts
from apertus_common.model import Assistant, Conversation, VerifiableAnswer


class VerifiableAnswerChecker(Checker):
    """Placement and cardinality of verifiable answers per burst.

    Warnings by default because the spec describes canonical placement and
    burst-vs-task scoping is not fully settled.
    """

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule(
            id="verifiable-answer/after-assistant",
            description=(
                "a verifiable_answer canonically precedes the assistant "
                "message of its burst"
            ),
            default_severity=Severity.WARNING,
        ),
        Rule(
            id="verifiable-answer/multiple",
            description="a burst carries at most one verifiable_answer",
            default_severity=Severity.WARNING,
        ),
    )

    def check(self, conversation: Conversation) -> Iterable[Violation]:
        for burst in iter_bursts(conversation):
            assistant_seen = False
            answers = 0
            for index in burst:
                item = conversation.items[index]
                if isinstance(item, Assistant):
                    assistant_seen = True
                elif isinstance(item, VerifiableAnswer):
                    answers += 1
                    if assistant_seen:
                        yield self.violation(
                            "verifiable-answer/after-assistant",
                            Location(item=index),
                            "verifiable_answer follows an assistant "
                            "message in the same burst",
                        )
                    if answers > 1:
                        yield self.violation(
                            "verifiable-answer/multiple",
                            Location(item=index),
                            f"burst carries {answers} verifiable_answer "
                            "messages; the task's parts belong in one",
                        )
