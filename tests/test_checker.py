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
    table = rule_table()
    assert len(table) == len({rule.id for rule in table})
    assert all(rule.description for rule in table)


def test_lightweight_paths_do_not_load_jsonschema():
    code = """
import sys
from apertus_common import Conversation, Reply, parse_output_text
Conversation(items=[Reply(payload="text")]).check(mode="profile")
parse_output_text('<|out|>call{"name":"weather","counter":0}<|hdr|>bad JSON<|/out|>')
assert 'jsonschema' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
