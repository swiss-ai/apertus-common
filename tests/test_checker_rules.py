"""Profile checks, explicit history evidence and local-only schema checks."""

from dataclasses import FrozenInstanceError

import pytest

from apertus_common.checker import (
    CallState,
    CheckContext,
    HistoryCheckpoint,
    HostClosure,
    check_arguments,
    parse_arguments,
)
from apertus_common.checker.rules import ArgumentsChecker
from apertus_common.errors import ArgumentError
from apertus_common.model import (
    Call,
    Claim,
    Conversation,
    Document,
    Host,
    Reply,
    Result,
    SystemPrompt,
    ToolSpec,
    UnknownOutput,
    User,
    Wait,
)


def tool(schema=None, name="weather"):
    return ToolSpec(
        name=name,
        description="Weather",
        json_schema=schema if schema is not None else {"type": "object"},
    )


def conversation(*items, tools=None):
    return Conversation(
        system=SystemPrompt.build(tools=tools if tools is not None else [tool()]),
        items=items,
    )


def call(counter=0, payload="{}", name="weather"):
    return Call(name=name, counter=counter, payload=payload)


def result(counter=0, name="weather"):
    return Result(name=name, counter=counter, payload="done")


def rules(report):
    return {finding.rule for finding in report.errors}


def test_profile_fields_and_document_provenance():
    conv = conversation(
        Document(payload="x"),
        UnknownOutput(type="quack", payload="x"),
        call(-1),
        tools=[tool(), tool()],
    )
    assert rules(conv.check("profile")) == {
        "profile/document-provenance",
        "profile/known-output",
        "profile/nonnegative-counter",
        "profile/unique-tool-name",
    }
    assert "profile/document-provenance" in rules(Document(payload="text").check())
    provenance = Document(payload="text", from_="user", id="d1").check()
    assert provenance.ok and provenance.complete
    assert "profile/object-schema" in rules(tool({"type": "array"}).check())


def test_serving_system_and_unknown_system():
    assert "conversation/system" in rules(Conversation().check())
    report = Conversation().check(context=CheckContext(history="partial"))
    assert report.ok and not report.complete


def test_pending_calls_partial_tail_bare_wait_and_parallel_results_are_valid():
    for items in [(call(),), (Wait(),), (call(), call(1), Wait(), result(1), result())]:
        report = conversation(*items).check()
        assert report.ok and report.complete, report


def test_counter_sequence_crosses_bursts_and_tools():
    conv = conversation(
        call(),
        Wait(),
        result(),
        call(1, name="other"),
        tools=[tool(), tool(name="other")],
    )
    assert conv.check().ok
    assert "conversation/counter" in rules(conversation(call(2)).check())
    assert "conversation/counter" in rules(conversation(call(), call()).check())


def test_tool_lookup_is_separate_from_arguments():
    report = conversation(call(name="missing")).check()
    assert "conversation/declared-tool" in rules(report)
    assert "arguments/json" not in rules(report)


@pytest.mark.parametrize(
    "items", [(result(),), (call(), result(name="other")), (call(), result(), result())]
)
def test_invalid_result_relationships(items):
    assert "conversation/result" in rules(conversation(*items).check())


def test_document_ids_and_wait_transitions():
    assert "conversation/document-id" in rules(
        conversation(
            Document(payload="a", from_="user", id="same"),
            Document(payload="b", from_="source", id="same"),
        ).check()
    )
    assert "conversation/wait" in rules(
        conversation(Wait(), Reply(payload="bad")).check()
    )
    assert (
        conversation(Wait(), User(payload="continue"), Reply(payload="good")).check().ok
    )
    assert conversation(Wait()).check().ok


def test_claim_placement_is_advisory_and_multiple_claims_allowed():
    report = conversation(
        Claim(payload="a"),
        Claim(payload="b"),
        Reply(payload="text"),
        Claim(payload="c"),
    ).check()
    assert report.ok and len(report.warnings) == 1


def test_prefix_supplies_system_counter_and_calls():
    prefix = conversation(call())
    fragment = Conversation(items=[result(), call(1)])
    report = fragment.check(context=CheckContext(prefix=prefix))
    assert report.ok and report.complete
    bad_prefix = conversation(call(9))
    report = fragment.check(context=CheckContext(prefix=bad_prefix))
    assert any(f.location.source == "prefix" for f in report.errors)


def test_partial_history_checks_local_facts_without_inventing_earlier_calls():
    report = conversation(result(4), call(5), call(6)).check(
        context=CheckContext(history="partial")
    )
    assert report.ok and not report.complete
    assert {f.rule for f in report.unevaluated} >= {
        "conversation/result",
        "conversation/counter",
    }
    duplicate = conversation(call(5), call(5)).check(
        context=CheckContext(history="partial")
    )
    assert "conversation/counter" in rules(duplicate)


def test_checkpoint_supplies_known_state():
    checkpoint = HistoryCheckpoint(
        system=SystemPrompt.build(tools=[tool()]),
        next_call_counter=4,
        calls=(CallState(3, "weather"),),
        document_ids=frozenset({"earlier"}),
        waiting=True,
    )
    context = CheckContext(history="partial", checkpoint=checkpoint)
    report = Conversation(items=[result(3), call(4)]).check(context=context)
    assert report.ok and report.complete
    duplicate_document = Conversation(
        items=[Document(payload="x", from_="user", id="earlier")]
    ).check(context=context)
    assert "conversation/document-id" in rules(duplicate_document)
    with pytest.raises(FrozenInstanceError):
        checkpoint.waiting = False


def test_host_closures_are_explicit_never_inferred_from_prose():
    conv = conversation(
        call(), Host(payload="NotCompleted: call 0 is closed."), result()
    )
    unknown = conv.check()
    assert unknown.ok and not unknown.complete
    assert {f.rule for f in unknown.unevaluated} == {
        "conversation/host-closure",
        "conversation/result",
    }
    no_closures = conv.check(context=CheckContext(host_closures=()))
    assert no_closures.ok and no_closures.complete
    known = conv.check(
        context=CheckContext(
            host_closures=(HostClosure(item=1, counter=0, name="weather"),)
        )
    )
    assert "conversation/result" in rules(known)
    assert known.complete


def test_context_contradictions_are_caller_errors():
    with pytest.raises(ValueError, match="conflicting system"):
        conversation().check(
            context=CheckContext(
                prefix=Conversation(system=SystemPrompt.build(behavior="different"))
            )
        )
    with pytest.raises(ValueError, match="retained host"):
        conversation(call()).check(
            context=CheckContext(
                host_closures=(HostClosure(item=0, counter=0, name="weather"),)
            )
        )
    with pytest.raises(ValueError, match="checkpoint requires"):
        CheckContext(checkpoint=HistoryCheckpoint())


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        "null",
        '{"a":1,"a":2}',
        '{"a":{"x":1,"x":2}}',
        '{"n":NaN}',
        '{"n":1e999}',
        '{"text":"\\ud800"}',
        '{"\\udfff":1}',
        '{"nested":[{"text":"\\ud800"}]}',
    ],
)
def test_strict_argument_json(payload):
    with pytest.raises(ArgumentError):
        parse_arguments(call(payload=payload))
    assert "arguments/json" in rules(check_arguments(call(payload=payload), tool()))
    assert "arguments/json" in rules(conversation(call(payload=payload)).check())


def test_argument_unicode_pair_and_fresh_mutable_result():
    message = call(payload='{"nested":[{"text":"\\ud83d\\ude42"}]}')
    arguments = parse_arguments(message)
    assert arguments == {"nested": [{"text": "🙂"}]}
    arguments["nested"][0]["text"] = "changed"
    assert parse_arguments(message) == {"nested": [{"text": "🙂"}]}


def test_argument_schema_without_coercion_default_insertion_or_format_checks():
    declaration = tool(
        {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 5},
                "email": {"type": "string", "format": "email"},
            },
            "required": ["n"],
        }
    )
    raw = call(payload=' {"n":"2"} ')
    assert "arguments/schema" in rules(check_arguments(raw, declaration))
    assert raw.payload == ' {"n":"2"} '
    assert check_arguments(
        call(payload='{"n":2,"email":"not an email"}'), declaration
    ).ok
    assert "arguments/schema" in rules(check_arguments(call(), declaration))


@pytest.mark.parametrize(
    "dialect",
    [
        None,
        "http://json-schema.org/draft-07/schema#",
        "https://json-schema.org/draft/2019-09/schema",
        "https://json-schema.org/draft/2020-12/schema",
    ],
)
def test_supported_dialects_and_internal_references(dialect):
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/count"}},
        "$defs": {"count": {"type": "integer"}},
    }
    if dialect is not None:
        schema["$schema"] = dialect
    assert check_arguments(call(payload='{"x":2}'), tool(schema)).ok
    assert "arguments/schema" in rules(
        check_arguments(call(payload='{"x":"two"}'), tool(schema))
    )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"x": {"type": "not-a-type"}}},
        {"type": "object", "$schema": "https://unknown.example/schema"},
        {"type": "object", "properties": {"unused": {"$ref": "#/$defs/missing"}}},
        {"type": "object", "$ref": "file:///etc/passwd"},
        {"type": "object", "$ref": "https://example.com/schema"},
    ],
)
def test_broken_even_unused_schema_is_reported(schema):
    report = conversation(tools=[tool(schema)]).check()
    assert "arguments/broken-schema" in rules(report)
    assert "arguments/schema" not in rules(report)


def test_instance_data_is_not_walked_as_schema():
    declaration = tool(
        {"type": "object", "examples": [{"$ref": "not-a-schema-reference"}]}
    )
    assert check_arguments(call(), declaration).ok


def test_observed_closures_remain_known_in_partial_histories():
    partial = CheckContext(history="partial")
    assert "conversation/result" in rules(
        conversation(result(5), result(5)).check(context=partial)
    )
    closures = CheckContext(
        history="partial",
        host_closures=(HostClosure(item=0, counter=5, name="weather"),),
    )
    assert "conversation/result" in rules(
        conversation(Host(payload="closed"), result(5)).check(context=closures)
    )


def test_schema_local_resources_and_anchors():
    declaration = tool(
        {
            "$id": "https://example.com/root",
            "type": "object",
            "$defs": {"number": {"$id": "number", "type": "integer"}},
            "properties": {"count": {"$ref": "number"}},
        }
    )
    assert check_arguments(call(payload='{"count":2}'), declaration).ok
    assert "arguments/schema" in rules(
        check_arguments(call(payload='{"count":"two"}'), declaration)
    )


def test_context_evidence_rejects_empty_names_and_document_ids():
    with pytest.raises(ValueError):
        CallState(0, "")
    with pytest.raises(ValueError):
        HostClosure(1, 0, "")
    with pytest.raises(ValueError):
        HistoryCheckpoint(document_ids=frozenset({""}))


def test_arguments_checker_reports_missing_matching_declaration():
    report = conversation(call(name="missing")).check(checkers=[ArgumentsChecker()])
    assert report.ok and not report.complete
    assert {finding.rule for finding in report.unevaluated} == {"arguments/schema"}


@pytest.mark.parametrize("keyword", ["properties", "$defs"])
def test_unsupported_nested_schema_dialect_is_reported_even_if_unused(keyword):
    declaration = tool(
        {
            "type": "object",
            keyword: {
                "unused": {
                    "$schema": "https://unknown.example/schema",
                    "type": "string",
                }
            },
        }
    )
    assert "arguments/broken-schema" in rules(conversation(tools=[declaration]).check())


@pytest.mark.parametrize("keyword", ["examples", "default", "const"])
def test_instance_dialect_keys_are_not_treated_as_schema_declarations(keyword):
    instance = {"$schema": "https://unknown.example/schema"}
    declaration = tool(
        {"type": "object", keyword: [instance] if keyword == "examples" else instance}
    )
    assert conversation(tools=[declaration]).check().ok


def test_wait_after_wait_requires_input():
    assert "conversation/wait" in rules(conversation(Wait(), Wait()).check())
    assert conversation(Wait(), User(payload="again"), Wait()).check().ok
    fragment = Conversation(items=[Wait()]).check(
        context=CheckContext(history="partial")
    )
    assert fragment.ok and not fragment.complete
