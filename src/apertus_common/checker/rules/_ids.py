"""The tool-call id convention.

The model conventionally writes ``id=TOOL_NAME:COUNTER`` with one
counter global to the conversation, and the harness echo is gated on the
``NAME:COUNTER`` shape. The accepted id grammar is wider while the
echoed shape is unsettled upstream, so every rule here defaults to
warning.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from apertus_common.checker._report import Location, Severity, Violation
from apertus_common.checker.rules._base import Checker, Rule
from apertus_common.model import Conversation, ToolCall, is_canonical_id


class IdConventionChecker(Checker):
    """The canonical ``NAME:COUNTER`` convention for model-written ids.

    Warnings by default: the id-echo shape is unsettled upstream, the
    name-part convention is never a MUST in words, and counters may start
    mid-conversation in a truncated transcript. Non-canonical ids skip the
    remaining sub-rules.
    """

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule(
            id="tool-call-id/non-canonical",
            description="a call id follows the NAME:COUNTER convention",
            default_severity=Severity.WARNING,
        ),
        Rule(
            id="tool-call-id/name-mismatch",
            description="the NAME part of a call id equals the called tool",
            default_severity=Severity.WARNING,
        ),
        Rule(
            id="tool-call-id/duplicate-counter",
            description="the counter is unique across the conversation",
            default_severity=Severity.WARNING,
        ),
        Rule(
            id="tool-call-id/non-sequential",
            description="the counter increments by one per call",
            default_severity=Severity.WARNING,
        ),
    )

    def check(self, conversation: Conversation) -> Iterable[Violation]:
        seen_counters: set[int] = set()
        previous_counter: int | None = None
        for index, item in enumerate(conversation.items):
            if not isinstance(item, ToolCall):
                continue
            if not is_canonical_id(item.id):
                yield self.violation(
                    "tool-call-id/non-canonical",
                    Location(item=index),
                    f"call id {item.id!r} does not follow NAME:COUNTER",
                )
                continue
            name, counter_text = item.id.rsplit(":", 1)
            counter = int(counter_text)
            if name != item.name:
                yield self.violation(
                    "tool-call-id/name-mismatch",
                    Location(item=index),
                    f"id names {name!r} but the call names {item.name!r}",
                )
            if counter in seen_counters:
                yield self.violation(
                    "tool-call-id/duplicate-counter",
                    Location(item=index),
                    f"counter {counter} was already used; the counter is "
                    "global to the conversation",
                )
            elif previous_counter is not None and counter != previous_counter + 1:
                yield self.violation(
                    "tool-call-id/non-sequential",
                    Location(item=index),
                    f"counter jumped from {previous_counter} to {counter}",
                )
            seen_counters.add(counter)
            previous_counter = counter
