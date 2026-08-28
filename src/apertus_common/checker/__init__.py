"""Conformance checking of conversations against the format spec.

``check()`` runs a strict default set of rule-family checkers over a
typed ``Conversation`` and reports violations instead of raising.
Extend by subclassing ``Checker`` and passing instances.
"""

from apertus_common.checker._check import check, default_checkers, rule_table
from apertus_common.checker._report import (
    ConformanceError,
    ConformanceReport,
    Location,
    Severity,
    Violation,
)
from apertus_common.checker.rules import (
    Checker,
    IdConventionChecker,
    InputOrderChecker,
    Rule,
    SystemPromptChecker,
    ToolCallChecker,
    ToolResultChecker,
    VerifiableAnswerChecker,
    WaitChecker,
)

__all__ = [
    "Checker",
    "ConformanceError",
    "ConformanceReport",
    "IdConventionChecker",
    "InputOrderChecker",
    "Location",
    "Rule",
    "Severity",
    "SystemPromptChecker",
    "ToolCallChecker",
    "ToolResultChecker",
    "VerifiableAnswerChecker",
    "Violation",
    "WaitChecker",
    "check",
    "default_checkers",
    "rule_table",
]
