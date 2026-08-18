"""Checker framework: report semantics, severity config, check()."""

from collections.abc import Iterable
from typing import ClassVar

import pytest

from apertus_common import (
    Checker,
    ConformanceReport,
    Conversation,
    Harness,
    Location,
    Rule,
    Severity,
    SystemPrompt,
    ToolCall,
    ToolSpec,
    User,
    Violation,
    Wait,
    check,
    default_checkers,
    rule_table,
)
from conftest import full_cast, pretraining


class _FlagChecker(Checker):
    """Test-local family with one error rule and one warning rule."""

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule("flag/hard", "always fires", Severity.ERROR),
        Rule("flag/soft", "always fires", Severity.WARNING),
    )

    def check(self, conversation: Conversation) -> Iterable[Violation]:
        yield self.violation("flag/hard", Location(item=0), "hard")
        yield self.violation("flag/soft", Location(item=0), "soft")


class _ConflictingChecker(Checker):
    """Test-local checker that deliberately reuses another family's id."""

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule("flag/hard", "conflicts with _FlagChecker", Severity.ERROR),
    )

    def check(self, conversation: Conversation) -> Iterable[Violation]:
        return ()


EXPECTED_RULE_SEVERITIES = {
    "system/duplicate-tool-name": Severity.ERROR,
    "input-order/canonical-order": Severity.ERROR,
    "tool-call/undeclared-name": Severity.ERROR,
    "tool-call/args-invalid": Severity.ERROR,
    "tool-call/broken-schema": Severity.ERROR,
    "tool-result/unknown-id": Severity.ERROR,
    "tool-result/duplicate": Severity.ERROR,
    "tool-result/open-call": Severity.WARNING,
    "tool-call-id/non-canonical": Severity.WARNING,
    "tool-call-id/name-mismatch": Severity.WARNING,
    "tool-call-id/duplicate-counter": Severity.WARNING,
    "tool-call-id/non-sequential": Severity.WARNING,
    "wait/unterminated-transcript": Severity.WARNING,
    "verifiable-answer/after-assistant": Severity.WARNING,
    "verifiable-answer/multiple": Severity.WARNING,
}


def test_report_semantics() -> None:
    error = Violation("flag/hard", Severity.ERROR, Location(item=0), "x")
    warning = Violation("flag/soft", Severity.WARNING, Location(item=1), "y")
    report = ConformanceReport(violations=(error, warning))
    assert not report.ok
    assert report.errors == (error,)
    assert report.warnings == (warning,)
    assert list(report) == [error, warning]
    assert len(report) == 2
    assert ConformanceReport(violations=(warning,)).ok


def test_location_rendering() -> None:
    assert str(Location(item=3)) == "items[3]"
    assert str(Location(item=8, field="args")) == "items[8].args"
    assert (
        str(Location(scope="system", field="tools[1].json_schema"))
        == "system.tools[1].json_schema"
    )
    assert str(Location(scope="system")) == "system"
    assert str(Location(scope="stream")) == "stream"
    with pytest.raises(ValueError):
        Location(scope="system", item=0)
    with pytest.raises(ValueError):
        Location(item=-1)


def test_severity_overrides() -> None:
    conversation = Conversation(items=[User(payload="x")])
    default = check(conversation, checkers=[_FlagChecker()])
    assert {v.rule: v.severity for v in default} == {
        "flag/hard": Severity.ERROR,
        "flag/soft": Severity.WARNING,
    }
    configured = check(
        conversation,
        checkers=[
            _FlagChecker(
                severity=Severity.WARNING,
                overrides={"flag/soft": Severity.ERROR},
            )
        ],
    )
    assert {v.rule: v.severity for v in configured} == {
        "flag/hard": Severity.WARNING,
        "flag/soft": Severity.ERROR,
    }
    call_level = check(
        conversation,
        checkers=[_FlagChecker(severity=Severity.WARNING)],
        overrides={"flag/soft": Severity.ERROR},
    )
    assert not call_level.ok
    assert call_level.errors[0].rule == "flag/soft"


def test_invalid_checker_configuration_rejected() -> None:
    with pytest.raises(ValueError):
        Rule("Bad Rule", "invalid", Severity.ERROR)
    with pytest.raises(ValueError):
        Rule("flag/empty", "", Severity.ERROR)
    with pytest.raises(ValueError):
        _FlagChecker(overrides={"flag/missing": Severity.WARNING})
    with pytest.raises(ValueError):
        check(
            Conversation(),
            checkers=[_FlagChecker()],
            overrides={"other/rule": Severity.WARNING},
        )
    with pytest.raises(KeyError):
        _FlagChecker().violation("flag/missing", Location(), "missing")
    with pytest.raises(ValueError, match="declared by both"):
        check(
            Conversation(),
            checkers=[_FlagChecker(), _ConflictingChecker()],
        )


def test_rule_table() -> None:
    table = rule_table()
    assert {rule.id: rule.default_severity for rule in table} == (
        EXPECTED_RULE_SEVERITIES
    )
    defaults = default_checkers()
    assert {rule.id for checker in defaults for rule in checker.rules} == set(
        EXPECTED_RULE_SEVERITIES
    )
    assert all(rule.description for rule in table)
    assert all(
        first is not second
        for first, second in zip(defaults, default_checkers(), strict=True)
    )


def test_default_set_on_full_cast() -> None:
    # full_cast() is deliberately non-conformant: GenericInput (order 3)
    # follows User (order 5), and get_weather:0 has no prior call.
    report = check(full_cast())
    assert not report.ok
    assert {v.rule for v in report.errors} == {
        "input-order/canonical-order",
        "tool-result/unknown-id",
    }
    normalized = check(full_cast().normalized())
    assert "input-order/canonical-order" not in {v.rule for v in normalized.errors}
    assert check(full_cast(), checkers=[]).ok  # an explicit set replaces defaults


def test_conforming_conversation_is_ok() -> None:
    conversation = Conversation(
        system=SystemPrompt(
            effort="low",
            tools=[ToolSpec(name="bash", json_schema={"type": "object"})],
        ),
        items=[
            Harness(payload="notice"),
            User(payload="run it"),
            ToolCall(id="bash:0", name="bash", args={}),
            Wait(),
        ],
    )
    report = check(conversation)
    assert report.ok
    assert {v.rule for v in report.warnings} == {"tool-result/open-call"}


def test_pretraining_is_clean() -> None:
    assert check(pretraining()).ok
    assert len(check(pretraining())) == 0
