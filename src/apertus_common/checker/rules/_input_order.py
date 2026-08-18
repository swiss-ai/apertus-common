"""Canonical input order at a boundary.

When multiple inputs are delivered at the same boundary, they appear in
canonical order: harness, tool_result, retrieval/event, attachment,
user. Types without a declared position (open vocabulary, ``document``)
key at 3, matching ``Conversation.normalized()``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from apertus_common.checker._report import Location, Severity, Violation
from apertus_common.checker.rules._base import Checker, Rule
from apertus_common.model import Conversation, InputMessage, delivery_order


class InputOrderChecker(Checker):
    """Canonical delivery order within each run of consecutive inputs.

    Error by default: the format spec states the order declaratively. One
    violation per adjacent decreasing pair, located at the later item.
    Shares the one order definition with ``Conversation.normalized()``.
    """

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule(
            id="input-order/canonical-order",
            description=("inputs at one boundary appear in canonical delivery order"),
            default_severity=Severity.ERROR,
        ),
    )

    def check(self, conversation: Conversation) -> Iterable[Violation]:
        previous_order: int | None = None
        for index, item in enumerate(conversation.items):
            if not isinstance(item, InputMessage):
                previous_order = None
                continue
            current_order = delivery_order(item)
            if previous_order is not None and current_order < previous_order:
                yield self.violation(
                    "input-order/canonical-order",
                    Location(item=index),
                    f"input of order {current_order} follows an input "
                    f"of order {previous_order} at the same boundary",
                )
            previous_order = current_order
