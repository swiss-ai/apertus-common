"""Checking remains explicit, immutable and separate from loading/parsing."""

import subprocess
import sys

import pytest

from apertus_common.checker import (
    Checker,
    CheckOptions,
    ConformanceError,
    Location,
    Rule,
    Severity,
    checkers_for,
    rule_table,
)
from apertus_common.model import (
    Call,
    Conversation,
    Document,
    Reply,
    SystemPrompt,
    ToolSpec,
)


def system():
    return SystemPrompt.build(
        tools=[
            ToolSpec(
                name="weather", description="Weather", json_schema={"type": "object"}
            )
        ]
    )


def test_presets_and_replacement():
    call = Call(name="weather", counter=-1, payload="bad JSON")
    conv = Conversation(system=system(), items=[call])
    assert {v.rule for v in conv.check("profile")} == {"profile/nonnegative-counter"}
    assert "arguments/json" not in {v.rule for v in conv.check("conversation")}
    assert "arguments/json" in {v.rule for v in conv.check("full")}
    assert conv.check(checkers=[]).ok
    assert len(checkers_for("profile")) < len(checkers_for("full"))
    assert len(checkers_for("full", training=True)) == len(checkers_for("full")) + 1
    with pytest.raises(ValueError, match="unknown check mode"):
        conv.check("typo")


def test_checked_loading_and_builder_use_same_options():
    options = CheckOptions(mode="conversation")
    conv = (
        Conversation.builder()
        .system(system())
        .document("text", from_="user", id="d1")
        .call(name="weather", counter=0, payload="not JSON")
        .build(check=options)
    )
    assert Conversation.from_json(conv.to_json(), check=options) == conv
    with pytest.raises(ConformanceError) as caught:
        Conversation.from_json(conv.to_json(), check="full")
    assert not caught.value.report.ok


def test_checked_loading_rejects_incomplete_even_without_errors():
    call = Call(name="weather", counter=7, payload="{}")
    with pytest.raises(ConformanceError) as caught:
        Call.from_json(call.to_json(), check="full")
    assert caught.value.report.ok
    assert not caught.value.report.complete


def test_checks_do_not_mutate_or_sort():
    conv = Conversation(
        system=system(), items=[Reply(payload="first"), Document(payload="second")]
    )
    before = conv.to_json()
    conv.check()
    assert conv.to_json() == before


def test_custom_checker_and_warning():
    class Advisory(Checker):
        rules = (Rule("custom/advice", "custom advisory", Severity.WARNING),)

        def check(self, value, context=None):
            yield self.violation(
                "custom/advice", Location(), "Consider reviewing this."
            )

    report = Conversation().check(checkers=[Advisory()])
    assert report.ok and report.complete and len(report.warnings) == 1
    assert list(report) == list(report.violations)
    assert len(report) == 1
    with pytest.raises(ValueError, match="unique"):
        Conversation().check(checkers=[Advisory(), Advisory()])


def test_rule_table_unique():
    from apertus_common.checker._report import RULE_ID_RE

    table = rule_table()
    assert len(table) == len({rule.id for rule in table})
    assert all(RULE_ID_RE.fullmatch(rule.id) for rule in table)
    assert all(rule.description.strip() for rule in table)
    assert all(checker.rules for checker in checkers_for("full", training=True))
    assert {"training/input-mid-burst", "training/final-wait"} <= {
        rule.id for rule in table
    }


def test_lightweight_paths_do_not_load_jsonschema():
    code = """
import sys
from apertus_common import Conversation, Reply, parse_output_text
Conversation(items=[Reply(payload="text")]).check(mode="profile")
parse_output_text('<|out|>call{"name":"weather","counter":0}<|hdr|>bad JSON<|/out|>')
assert 'jsonschema' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_builtin_checks_share_one_view_per_value(monkeypatch):
    from apertus_common.checker import _check
    from apertus_common.checker._report import ConformanceReport, Unevaluated, Violation

    prepared = []
    original = _check.view

    def observe(value, context):
        prepared.append(value)
        return original(value, context)

    monkeypatch.setattr(_check, "view", observe)
    checks = checkers_for("full", training=True)
    for value in (
        Conversation(),
        Conversation(system=system(), items=[Reply(payload="x")]),
    ):
        findings = [finding for checker in checks for finding in checker.check(value)]
        expected = ConformanceReport(
            tuple(f for f in findings if isinstance(f, Violation)),
            tuple(f for f in findings if isinstance(f, Unevaluated)),
        )
        assert value.check(checkers=checks) == expected
    assert len(prepared) == 2 and prepared[0] is not prepared[1]


@pytest.mark.parametrize("instance_override", [False, True])
def test_custom_builtin_override_keeps_control_of_checking(
    monkeypatch, instance_override
):
    from apertus_common import CheckContext, ProfileChecker

    called = []

    class CustomProfile(ProfileChecker):
        def check(self, value, context=None):
            called.append((value, context))
            return iter(())

    value = Conversation(system=SystemPrompt.build(behavior="one"))
    context = CheckContext(
        prefix=Conversation(system=SystemPrompt.build(behavior="two"))
    )
    checker = CustomProfile()
    if instance_override:
        wrapped = ProfileChecker()
        monkeypatch.setattr(wrapped, "check", checker.check)
        checker = wrapped
    # Custom checks may deliberately ignore evidence that built-ins would reject.
    assert value.check(checkers=[checker], context=context).ok
    assert called == [(value, context)]
    assert value.check(checkers=[], context=context).ok
