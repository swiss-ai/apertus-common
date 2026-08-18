"""Tool result echo discipline.

A tool result answers a pending tool call: the harness echoes the
model's id from the call verbatim, and every call eventually receives
exactly one result, which closes it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from apertus_common.checker._report import Location, Severity, Violation
from apertus_common.checker.rules._base import Checker, Rule
from apertus_common.model import Conversation, ToolCall, ToolResult


class ToolResultChecker(Checker):
    """Each result closes exactly one prior open call, byte-exactly.

    Errors by default for unknown ids and duplicates, which the format
    spec requires. A
    call still open at the end of the transcript is a warning only: the
    spec allows calls to stay open across resumptions ("the remaining
    calls simply stay open").
    """

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule(
            id="tool-result/unknown-id",
            description="a result echoes the id of a pending call",
            default_severity=Severity.ERROR,
        ),
        Rule(
            id="tool-result/duplicate",
            description="exactly one result closes a call",
            default_severity=Severity.ERROR,
        ),
        Rule(
            id="tool-result/open-call",
            description="a call is still open at the end of the transcript",
            default_severity=Severity.WARNING,
        ),
    )

    def check(self, conversation: Conversation) -> Iterable[Violation]:
        open_calls: dict[str, list[int]] = {}
        closed: set[str] = set()
        for index, item in enumerate(conversation.items):
            if isinstance(item, ToolCall):
                open_calls.setdefault(item.id, []).append(index)
            elif isinstance(item, ToolResult):
                pending = open_calls.get(item.id)
                if pending:
                    pending.pop(0)
                    if not pending:
                        del open_calls[item.id]
                        closed.add(item.id)
                elif item.id in closed:
                    yield self.violation(
                        "tool-result/duplicate",
                        Location(item=index),
                        f"call {item.id!r} was already closed by an earlier result",
                    )
                else:
                    yield self.violation(
                        "tool-result/unknown-id",
                        Location(item=index),
                        f"result id {item.id!r} matches no pending tool call",
                    )
        for call_id, indices in open_calls.items():
            for call_index in indices:
                yield self.violation(
                    "tool-result/open-call",
                    Location(item=call_index),
                    f"call {call_id!r} never received its result",
                )
