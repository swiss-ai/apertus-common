"""Optional conformance checking; model methods are the canonical entry point."""

from apertus_common.checker._arguments import check_arguments, parse_arguments
from apertus_common.checker._check import (
    apply_check as apply_check,
)
from apertus_common.checker._check import (
    check_value as check_value,
)
from apertus_common.checker._check import (
    checkers_for,
    rule_table,
)
from apertus_common.checker._context import (
    CallState,
    CheckContext,
    CheckMode,
    CheckOptions,
    HistoryCheckpoint,
    HostClosure,
)
from apertus_common.checker._report import (
    ConformanceError,
    ConformanceReport,
    Location,
    Severity,
    Unevaluated,
    Violation,
)
from apertus_common.checker.rules import (
    ArgumentsChecker,
    Checker,
    ConversationChecker,
    ProfileChecker,
    Rule,
)

__all__ = [
    "ArgumentsChecker",
    "CallState",
    "CheckContext",
    "CheckMode",
    "CheckOptions",
    "Checker",
    "ConformanceError",
    "ConformanceReport",
    "ConversationChecker",
    "HistoryCheckpoint",
    "HostClosure",
    "Location",
    "ProfileChecker",
    "Rule",
    "Severity",
    "Unevaluated",
    "Violation",
    "check_arguments",
    "checkers_for",
    "parse_arguments",
    "rule_table",
]
