"""Halt states and the wait token.

The wait token is the model's only deliberate stop; the engine halts
decoding at this token. Bare and consecutive waits are legal remnants of
think-stripping, waits never wait for open calls, and inputs
may land at any message boundary, so the only solid transcript rule is
completeness: a trailing output run without a terminating wait.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from apertus_common.checker._report import Location, Severity, Violation
from apertus_common.checker.rules._base import Checker, Rule
from apertus_common.model import Conversation, InputMessage, OutputMessage, Wait


class WaitChecker(Checker):
    """A transcript's final output run should end in a wait.

    Warning by default: an unterminated tail may be an incomplete
    transcript rather than a violation. The checker is silent when the
    conversation contains no wait at all, which covers pretraining
    sequences without a mode flag.
    """

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule(
            id="wait/unterminated-transcript",
            description="the final output run is not terminated by a wait",
            default_severity=Severity.WARNING,
        ),
    )

    def check(self, conversation: Conversation) -> Iterable[Violation]:
        items = conversation.items
        if not any(isinstance(item, Wait) for item in items):
            return
        # trailing spliced inputs do not terminate an output run either
        last = len(items) - 1
        while last >= 0 and isinstance(items[last], InputMessage):
            last -= 1
        if last >= 0 and isinstance(items[last], OutputMessage):
            yield self.violation(
                "wait/unterminated-transcript",
                Location(item=last),
                "the transcript ends inside an output run; generation "
                "stops only at the wait token",
            )
