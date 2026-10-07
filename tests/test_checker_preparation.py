"""Reusable checking preserves findings while avoiding repeated compilation."""

from unittest.mock import patch

import pytest

from apertus_common import (
    ArgumentsChecker,
    Call,
    CallState,
    CheckContext,
    CheckOptions,
    Conversation,
    HistoryCheckpoint,
    Host,
    ProfileChecker,
    Reply,
    Result,
    SystemPrompt,
    ToolSpec,
    Wait,
    checkers_for,
)
from apertus_common.checker._arguments import compile_schema


def declaration(schema, name="tool"):
    return ToolSpec(name=name, description="A tool", json_schema=schema)


def conversation(schema, payload="{}", name="tool"):
    return Conversation(
        system=SystemPrompt.build(tools=[declaration(schema, name)]),
        items=[Call(name=name, counter=0, payload=payload)],
    )


def test_cache_reuses_schema_but_not_call_findings():
    checker = ArgumentsChecker()
    schema = {"type": "object", "required": ["x"]}
    with patch(
        "apertus_common.checker.rules._arguments.compile_schema", wraps=compile_schema
    ) as compile_mock:
        for name, payload in [("a", "{}"), ("b", '{"x": 1}'), ("a", "invalid")]:
            conv = conversation(schema, payload, name)
            assert conv.check(checkers=[checker]) == conv.check(
                checkers=[ArgumentsChecker(schema_cache_size=0)]
            )
        # One cached compilation plus three uncached comparison runs.
        assert compile_mock.call_count == 4


def test_cached_schema_errors_keep_current_location_and_context():
    bad = conversation({"type": "object", "properties": {"x": {"type": "bad"}}})
    checker = ArgumentsChecker()
    with patch(
        "apertus_common.checker.rules._arguments.compile_schema", wraps=compile_schema
    ) as compile_mock:
        direct = bad.check(checkers=[checker])
        contextual = Conversation().check(
            checkers=[checker], context=CheckContext(prefix=bad)
        )
        assert compile_mock.call_count == 1
    assert direct.errors[0].rule == "arguments/broken-schema"
    assert direct.errors[0].location.source == "value"
    assert contextual.errors[0].location.source == "prefix"
    assert direct.errors[0].message == contextual.errors[0].message


def test_cache_is_bounded_lru_and_can_be_disabled():
    a = conversation({"type": "object"})
    b = conversation({"type": "object", "required": ["x"]})
    c = conversation({"type": "object", "required": ["y"]})
    checker = ArgumentsChecker(schema_cache_size=2)
    with patch(
        "apertus_common.checker.rules._arguments.compile_schema", wraps=compile_schema
    ) as compile_mock:
        for conv in [a, b, a, c, a, b]:
            conv.check(checkers=[checker])
        assert compile_mock.call_count == 4
    for size in [-1, True, 1.5]:
        with pytest.raises(ValueError, match="schema_cache_size"):
            ArgumentsChecker(schema_cache_size=size)


def test_cache_keys_include_dialect_and_preserve_schema_order():
    checker = ArgumentsChecker()
    schemas = [
        {"type": "object", "required": ["a", "b"]},
        {"required": ["a", "b"], "type": "object"},
        {"$schema": "http://json-schema.org/draft-07/schema", "type": "object"},
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    ]
    with patch(
        "apertus_common.checker.rules._arguments.compile_schema", wraps=compile_schema
    ) as compile_mock:
        for schema in schemas * 2:
            conversation(schema).check(checkers=[checker])
        assert compile_mock.call_count == len(schemas)


def test_loss_weight_opt_in_includes_all_sources_and_checked_loading():
    system = SystemPrompt.build(loss_weight=0)
    conv = Conversation(
        system=system, items=[Reply(payload="x", loss_weight=0.5), Wait(loss_weight=0)]
    )
    report = conv.check()
    assert [v.rule for v in report.errors] == ["profile/loss-weight"] * 3
    enabled = checkers_for("full", allow_loss_weights=True)
    assert conv.check(checkers=enabled).ok
    assert (
        Conversation.from_json(conv.to_json(), check=CheckOptions(checkers=enabled))
        == conv
    )
    assert conv.check(checkers=[ProfileChecker(allow_loss_weights=True)]).ok

    # Different training weights are not conflicting system configuration.
    ctx = CheckContext(
        history="partial",
        prefix=Conversation(system=SystemPrompt.build(loss_weight=0.25)),
        checkpoint=HistoryCheckpoint(system=SystemPrompt.build(loss_weight=0.75)),
    )
    findings = conv.check("profile", context=ctx).errors
    assert [v.location.source for v in findings[:3]] == [
        "checkpoint",
        "prefix",
        "value",
    ]
    assert conv.check(
        checkers=[ProfileChecker(allow_loss_weights=True)], context=ctx
    ).ok


def test_hosts_only_make_preceding_pending_calls_uncertain():
    tool = declaration({"type": "object"})
    conv = Conversation(
        system=SystemPrompt.build(tools=[tool]),
        items=[
            Call(name="tool", counter=0, payload="{}"),
            Host(payload="notice"),
            Host(payload="another"),
            Call(name="tool", counter=1, payload="{}"),
            Result(name="tool", counter=1, payload="done"),
            Result(name="tool", counter=0, payload="done"),
            Host(payload="all closed"),
        ],
    )
    report = conv.check()
    assert report.ok
    assert [(v.rule, v.location.item) for v in report.unevaluated] == [
        ("conversation/host-closure", 1),
        ("conversation/host-closure", 2),
        ("conversation/result", 5),
    ]
    # Pending state supplied by a checkpoint is tracked in the same way.
    fragment = Conversation(
        items=[
            Host(payload="notice"),
            Result(name="tool", counter=0, payload="done"),
            Host(payload="closed"),
        ]
    )
    context = CheckContext(
        history="partial",
        checkpoint=HistoryCheckpoint(
            system=SystemPrompt.build(tools=[tool]),
            next_call_counter=1,
            calls=(CallState(0, "tool"),),
            waiting=False,
        ),
    )
    assert [
        (v.rule, v.location.item) for v in fragment.check(context=context).unevaluated
    ] == [
        ("conversation/host-closure", 0),
        ("conversation/result", 1),
    ]
