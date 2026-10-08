"""Opt-in training-data advice for valid but usually unintended burst boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from apertus_common.checker._context import CheckContext
from apertus_common.checker._report import Severity, Unevaluated, Violation
from apertus_common.checker.rules._base import Checker, CheckValue, Rule
from apertus_common.checker.rules._view import view
from apertus_common.model import InputMessage, OutputMessage


class TrainingChecker(Checker):
    """Warn where turn-based training data leaves an output burst without a wait.

    Input during a burst and unfinished histories are valid Apertus 2, so these
    findings are warnings outside the conformance presets. Only observed adjacent
    items, including a supplied prefix, are compared.
    """

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule(
            "training/input-mid-burst",
            "training data closes an output burst with a wait before the next input",
            Severity.WARNING,
        ),
        Rule(
            "training/final-wait",
            "training data closes its final output burst with a wait",
            Severity.WARNING,
        ),
    )

    def check(
        self, value: CheckValue, context: CheckContext | None = None
    ) -> Iterable[Violation | Unevaluated]:
        current = view(value, context)
        if current.standalone:
            return
        previous = None
        for item, location in current.items:
            if isinstance(item, InputMessage) and isinstance(previous, OutputMessage):
                yield self.violation(
                    "training/input-mid-burst",
                    location,
                    "input follows output without a wait",
                )
            previous = item
        if isinstance(previous, OutputMessage):
            yield self.violation(
                "training/final-wait",
                current.items[-1][1],
                "the history ends with output, so the model does not learn to stop",
            )
