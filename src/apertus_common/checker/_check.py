"""Model-method delegation and named checker sets."""

from __future__ import annotations

from collections.abc import Sequence

from apertus_common.checker._context import CheckContext, CheckMode, CheckOptions
from apertus_common.checker._report import (
    ConformanceError,
    ConformanceReport,
    Unevaluated,
    Violation,
)
from apertus_common.checker.rules._arguments import ArgumentsChecker
from apertus_common.checker.rules._base import Checker, CheckValue, Rule
from apertus_common.checker.rules._conversation import ConversationChecker
from apertus_common.checker.rules._profile import ProfileChecker
from apertus_common.checker.rules._training import TrainingChecker


def checkers_for(
    mode: CheckMode, *, allow_loss_weights: bool = False, training: bool = False
) -> tuple[Checker, ...]:
    """Create reusable checkers; explicit checker lists replace this preset.

    `training=True` appends the advisory `TrainingChecker` for training records.
    """
    profile = ProfileChecker(allow_loss_weights=allow_loss_weights)
    match mode:
        case "profile":
            checkers: tuple[Checker, ...] = (profile,)
        case "conversation":
            checkers = (profile, ConversationChecker())
        case "full":
            checkers = (profile, ConversationChecker(), ArgumentsChecker())
        case _:
            raise ValueError(f"unknown check mode: {mode!r}")
    return (*checkers, TrainingChecker()) if training else checkers


def rule_table() -> tuple[Rule, ...]:
    """All built-in rules, including the opt-in training advice."""
    return tuple(
        rule
        for checker in checkers_for("full", training=True)
        for rule in checker.rules
    )


def check_value(
    value: CheckValue,
    mode: CheckMode = "full",
    *,
    checkers: Sequence[Checker] | None = None,
    context: CheckContext | None = None,
) -> ConformanceReport:
    active = checkers_for(mode) if checkers is None else tuple(checkers)
    # Even an explicit checker set must not silently accept a misspelled mode.
    if mode not in ("profile", "conversation", "full"):
        raise ValueError(f"unknown check mode: {mode!r}")
    declared: set[str] = set()
    violations: list[Violation] = []
    unevaluated: list[Unevaluated] = []
    for checker in active:
        ids = [rule.id for rule in checker.rules]
        if len(ids) != len(set(ids)) or declared.intersection(ids):
            raise ValueError("checker rule IDs must be unique within a run")
        declared.update(ids)
        for finding in checker.check(value, context):
            if finding.rule not in ids:
                raise ValueError(f"checker emitted undeclared rule {finding.rule!r}")
            if isinstance(finding, Violation):
                violations.append(finding)
            else:
                unevaluated.append(finding)
    return ConformanceReport(tuple(violations), tuple(unevaluated))


def apply_check[T: CheckValue](value: T, option: CheckMode | CheckOptions | None) -> T:
    """One shared opt-in path for native loading and builders."""
    if option is None:
        return value
    options = CheckOptions(mode=option) if isinstance(option, str) else option
    report = check_value(
        value, options.mode, checkers=options.checkers, context=options.context
    )
    if not report.ok or not report.complete:
        raise ConformanceError(report)
    return value
