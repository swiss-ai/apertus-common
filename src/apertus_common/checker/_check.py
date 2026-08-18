"""The public conformance entry point: ``check()`` and the rule table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from apertus_common.checker._report import ConformanceReport, Severity, Violation
from apertus_common.checker.rules._base import Checker, Rule
from apertus_common.checker.rules._ids import IdConventionChecker
from apertus_common.checker.rules._input_order import InputOrderChecker
from apertus_common.checker.rules._system_prompt import SystemPromptChecker
from apertus_common.checker.rules._tool_calls import ToolCallChecker
from apertus_common.checker.rules._tool_results import ToolResultChecker
from apertus_common.checker.rules._verifiable_answer import VerifiableAnswerChecker
from apertus_common.checker.rules._waits import WaitChecker
from apertus_common.model import Conversation

_BUILTIN_CHECKER_TYPES: tuple[type[Checker], ...] = (
    SystemPromptChecker,
    InputOrderChecker,
    ToolCallChecker,
    ToolResultChecker,
    IdConventionChecker,
    WaitChecker,
    VerifiableAnswerChecker,
)


def default_checkers() -> tuple[Checker, ...]:
    """Fresh default-configured instances of every builtin checker."""
    return tuple(checker_type() for checker_type in _BUILTIN_CHECKER_TYPES)


def rule_table() -> tuple[Rule, ...]:
    """Every rule declared by the builtin checkers."""
    return tuple(
        rule for checker_type in _BUILTIN_CHECKER_TYPES for rule in checker_type.rules
    )


def _declared_rules(checkers: Sequence[Checker]) -> set[str]:
    declared: set[str] = set()
    families: dict[str, type[Checker]] = {}
    for checker in checkers:
        for rule in checker.rules:
            owner = families.setdefault(rule.id, type(checker))
            if owner is not type(checker):
                raise ValueError(
                    f"rule id {rule.id!r} is declared by both "
                    f"{owner.__name__} and {type(checker).__name__}; "
                    "rule families are a namespace, pick a distinct one"
                )
            declared.add(rule.id)
    return declared


def check(
    conversation: Conversation,
    *,
    checkers: Sequence[Checker] | None = None,
    overrides: Mapping[str, Severity] | None = None,
) -> ConformanceReport:
    """Run conformance checkers over *conversation* and report violations.

    ``checkers=None`` runs :func:`default_checkers`; an explicit sequence
    replaces the set entirely (extend the defaults with
    ``[*default_checkers(), MyChecker()]``). ``overrides`` remaps rule
    ids to severities for this call; ids not declared by any active
    checker raise ``ValueError``, as do rule ids declared by two distinct
    checker types. Violations keep emission order; the report never
    raises on violations.
    """
    active = tuple(default_checkers()) if checkers is None else tuple(checkers)
    declared = _declared_rules(active)
    if overrides:
        unknown = sorted(set(overrides) - declared)
        if unknown:
            raise ValueError(
                f"override(s) for rule(s) {unknown} match no active checker"
            )
    violations: list[Violation] = []
    for checker in active:
        violations.extend(checker.check(conversation))
    if overrides:
        violations = [
            replace(violation, severity=overrides[violation.rule])
            if violation.rule in overrides
            else violation
            for violation in violations
        ]
    return ConformanceReport(violations=tuple(violations))
