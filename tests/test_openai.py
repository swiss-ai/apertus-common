"""Contracts for the explicit, loss-aware OpenAI compatibility subset."""

from typing import Any

import pytest
from pydantic import ValidationError

from apertus_common.errors import AdapterError
from apertus_common.formats import (
    MappingPolicy,
    OpenAIMappingContext,
    ToolAssociation,
    from_openai,
    to_openai,
    to_openai_output,
)
from apertus_common.model import (
    Call,
    Claim,
    Conversation,
    Document,
    Host,
    Reply,
    Result,
    SystemPrompt,
    Think,
    Thinking,
    ToolSpec,
    UnknownOutput,
    User,
    Wait,
)


def call_record(
    call_id: str = "external", name: str = "weather", arguments: Any = '{"city":"Bern"}'
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def test_request_round_trip_preserves_tools_arguments_and_external_ids() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Weather.",
                "parameters": {"type": "object"},
            },
        }
    ]
    request = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Bern?"},
        {
            **call_record(arguments=' {"city": "Bern"} '),
            "content": "Checking.",
            "reasoning_content": "Need weather.",
        },
        {"role": "tool", "tool_call_id": "external", "content": "Sunny."},
        {"role": "assistant", "content": "Sunny in Bern."},
    ]
    mapped = from_openai(request, tools=tools, reasoning_effort="high")
    assert mapped.value.system is not None
    assert mapped.value.system.thinking == Thinking.HIGH
    assert [type(item) for item in mapped.value.items] == [
        User,
        Think,
        Reply,
        Call,
        Wait,
        Result,
        Reply,
        Wait,
    ]
    restored = to_openai(mapped.value, context=mapped.context)
    # OpenAI permits omitted content; export consistently emits content=None.
    assert restored.value == {
        "messages": request,
        "tools": tools,
        "reasoning_effort": "high",
    }
    assert not mapped.diagnostics and not restored.diagnostics


@pytest.mark.parametrize(
    "raw", ["not json", "[]", '{"x":', '{"x":NaN}', '{"x":1,"x":2}']
)
def test_mapping_preserves_invalid_arguments_without_semantic_validation(
    raw: str,
) -> None:
    mapped = from_openai([call_record(arguments=raw)])
    assert mapped.value.items[0] == Call(name="weather", counter=0, payload=raw)
    output = to_openai_output(
        Conversation(items=mapped.value.items), context=mapped.context
    )
    assert output.value["tool_calls"][0]["function"]["arguments"] == raw


@pytest.mark.parametrize(
    "record",
    [
        {"role": "user", "content": "x", "name": "ignored"},
        {"role": "user", "content": [{"type": "image_url", "image_url": "x"}]},
        {"role": "user", "content": [{"type": "text", "text": "x", "ignored": True}]},
        {"role": "assistant", "content": "x", "refusal": "no"},
        {"role": "function", "content": "x"},
        {"role": "assistant", "tool_calls": None},
        call_record(arguments={}),
        call_record(call_id=""),
    ],
)
def test_unsupported_input_never_silently_drops_data(record: dict[str, Any]) -> None:
    with pytest.raises(AdapterError):
        from_openai([record])


def test_missing_arguments_and_unknown_tool_fields_rejected() -> None:
    record = call_record()
    del record["tool_calls"][0]["function"]["arguments"]
    with pytest.raises(AdapterError):
        from_openai([record])
    record = call_record()
    record["tool_calls"][0]["function"]["strict"] = True
    with pytest.raises(AdapterError):
        from_openai([record])
    with pytest.raises(AdapterError):
        from_openai(
            [],
            tools=[
                {"type": "function", "function": {"name": "weather", "strict": True}}
            ],
        )


def test_documented_normalizations_report_text_joining() -> None:
    mapped = from_openai(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ],
        tools=[{"type": "function", "function": {"name": "weather"}}],
    )
    assert mapped.value.items == (User(payload="ab"),)
    assert mapped.value.system is not None
    assert mapped.value.system.tools[0].description == ""
    assert mapped.value.system.tools[0].json_schema == {
        "type": "object",
        "properties": {},
    }
    assert mapped.diagnostics and not mapped.diagnostics[0].lossy


def test_system_merging_requires_specific_policy_and_preserves_order() -> None:
    messages = [
        {"role": "system", "content": "one"},
        {"role": "developer", "content": "two"},
    ]
    with pytest.raises(AdapterError):
        from_openai(messages)
    with pytest.raises(AdapterError):
        from_openai(messages[1:])
    mapped = from_openai(messages, policy=MappingPolicy(merge_system_messages=True))
    assert mapped.value.system is not None
    assert mapped.value.system.behavior == "one\n\ntwo"
    assert any(d.lossy for d in mapped.diagnostics)
    with pytest.raises(AdapterError):
        from_openai(
            [{"role": "user", "content": "x"}, *messages],
            policy=MappingPolicy(merge_system_messages=True),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning_effort": "high", "system_defaults": {"thinking": "low"}},
        {"system_defaults": {"behavior": "different"}},
        {
            "tools": [],
            "system_defaults": {
                "tools": [
                    ToolSpec(
                        name="weather", description="", json_schema={"type": "object"}
                    )
                ]
            },
        },
    ],
)
def test_conflicting_system_sources_are_errors(kwargs: dict[str, Any]) -> None:
    with pytest.raises(AdapterError):
        from_openai([{"role": "system", "content": "instructions"}], **kwargs)


def test_matching_defaults_and_explicit_values_are_accepted() -> None:
    mapped = from_openai(
        [{"role": "system", "content": "same"}],
        reasoning_effort="high",
        system_defaults={"thinking": Thinking.HIGH, "behavior": "same"},
    )
    assert mapped.value.system == SystemPrompt(thinking=Thinking.HIGH, behavior="same")


@pytest.mark.parametrize(
    "system",
    [
        SystemPrompt.build(identity="Apertus"),
        SystemPrompt.build(environment="sandbox"),
        SystemPrompt.build(
            tools=[
                ToolSpec(
                    name="weather", description="", json_schema={}, policy="ask first"
                )
            ]
        ),
    ],
)
def test_native_system_metadata_cannot_be_silently_flattened(
    system: SystemPrompt,
) -> None:
    with pytest.raises(AdapterError):
        to_openai(Conversation(system=system))


def test_context_survives_json_replay_and_compaction() -> None:
    first = from_openai([call_record("first")])
    context = OpenAIMappingContext.model_validate_json(first.context.model_dump_json())
    replay = from_openai([call_record("first")], context=context)
    assert replay.context == context and replay.value.items == first.value.items
    compacted = from_openai(
        [
            {"role": "tool", "tool_call_id": "first", "content": "done"},
            call_record("second"),
        ],
        context=context,
    )
    assert compacted.value.items[0] == Result(name="weather", counter=0, payload="done")
    assert compacted.value.items[1] == Call(
        name="weather", counter=1, payload='{"city":"Bern"}'
    )
    assert compacted.context.next_counter == 2
    assert context.next_counter == 1  # Mapping never mutates caller context.
    with pytest.raises(AdapterError):
        from_openai([call_record("first", name="other")], context=context)


def test_results_resolve_out_of_order_but_never_guess_associations() -> None:
    mapped = from_openai(
        [
            call_record("a"),
            call_record("b"),
            {"role": "tool", "tool_call_id": "b", "content": "B"},
            {"role": "tool", "tool_call_id": "a", "content": "A"},
        ]
    )
    assert mapped.value.items[-2:] == (
        Result(name="weather", counter=1, payload="B"),
        Result(name="weather", counter=0, payload="A"),
    )
    with pytest.raises(AdapterError):
        from_openai([{"role": "tool", "tool_call_id": "missing", "content": "x"}])
    with pytest.raises(AdapterError):
        to_openai(Conversation(items=[Result(name="weather", counter=0, payload="x")]))
    with pytest.raises(AdapterError):
        from_openai([call_record(), call_record()])


def test_native_ids_are_deterministic_and_collision_checked() -> None:
    context = OpenAIMappingContext(
        associations=(ToolAssociation(id="apertus_1", name="weather", counter=0),),
        next_counter=1,
    )
    conversation = Conversation(
        items=[
            Call(name="weather", counter=1, payload="{}"),
            Wait(),
            Result(name="weather", counter=1, payload="done"),
        ]
    )
    mapped = to_openai(conversation, context=context)
    allocated = mapped.value["messages"][0]["tool_calls"][0]["id"]
    assert allocated != "apertus_1"
    assert mapped.value["messages"][1]["tool_call_id"] == allocated
    assert to_openai(conversation, context=context).value == mapped.value
    with pytest.raises(AdapterError):
        to_openai(
            Conversation(items=[Call(name="other", counter=0, payload="{}"), Wait()]),
            context=context,
        )


@pytest.mark.parametrize(
    "associations,next_counter",
    [
        (
            [
                ToolAssociation(id="x", name="a", counter=0),
                ToolAssociation(id="x", name="a", counter=1),
            ],
            2,
        ),
        (
            [
                ToolAssociation(id="x", name="a", counter=0),
                ToolAssociation(id="y", name="a", counter=0),
            ],
            1,
        ),
        ([ToolAssociation(id="x", name="a", counter=1)], 1),
    ],
)
def test_inconsistent_context_rejected(
    associations: list[ToolAssociation], next_counter: int
) -> None:
    with pytest.raises(ValidationError):
        OpenAIMappingContext(associations=associations, next_counter=next_counter)


def test_history_splits_supported_segments_without_losing_message_order() -> None:
    native = Conversation(
        items=[Reply(payload="a"), Think(payload="b"), Reply(payload="c"), Wait()]
    )
    mapped = to_openai(native)
    assert from_openai(mapped.value["messages"]).value.items == native.items
    with pytest.raises(AdapterError):
        to_openai_output(native)
    merged = to_openai_output(
        native, policy=MappingPolicy(merge_assistant_messages=True)
    )
    assert merged.value["content"] == "ac"
    assert merged.value["reasoning_content"] == "b"
    assert any(d.lossy for d in merged.diagnostics)


@pytest.mark.parametrize(
    "items",
    [
        [Wait()],
        [Reply(payload="a")],
        [Reply(payload="a"), User(payload="queued"), Reply(payload="b"), Wait()],
        [Reply(payload="a"), Wait(), Reply(payload="b"), Wait()],
    ],
)
def test_unrepresentable_history_boundaries_require_explicit_policy(
    items: list[Any],
) -> None:
    native = Conversation(items=items)
    with pytest.raises(AdapterError):
        to_openai(native)
    mapped = to_openai(native, policy=MappingPolicy(merge_assistant_messages=True))
    assert any(d.lossy for d in mapped.diagnostics)


@pytest.mark.parametrize(
    "item",
    [
        Host(payload="x"),
        Document(payload="x"),
        UnknownOutput(output_type="future", payload="x"),
    ],
)
def test_no_generic_loss_policy_for_unsupported_native_messages(item: Any) -> None:
    with pytest.raises(AdapterError):
        to_openai(
            Conversation(items=[item]),
            policy=MappingPolicy(
                merge_assistant_messages=True,
                omit_claims=True,
                merge_system_messages=True,
            ),
        )


def test_claim_omission_requires_its_own_policy() -> None:
    native = Conversation(
        items=[Claim(payload="claim"), Reply(payload="answer"), Wait()]
    )
    with pytest.raises(AdapterError):
        to_openai_output(native, policy=MappingPolicy(merge_assistant_messages=True))
    mapped = to_openai_output(native, policy=MappingPolicy(omit_claims=True))
    assert mapped.value["content"] == "answer"
    assert any(d.code == "claim-omitted" and d.lossy for d in mapped.diagnostics)


def test_empty_external_function_name_is_a_mapping_error() -> None:
    with pytest.raises(AdapterError):
        from_openai([call_record(name="")])


def test_empty_assistant_message_maps_to_empty_reply_with_diagnostic() -> None:
    mapped = from_openai(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": None}]
    )
    assert mapped.value.items[1] == Reply(payload="")
    assert [d.code for d in mapped.diagnostics] == ["empty-assistant"]


def test_null_tool_fields_are_absent_and_strict_false_is_accepted() -> None:
    mapped = from_openai(
        [],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": None,
                    "parameters": None,
                    "strict": False,
                },
            }
        ],
    )
    assert mapped.value.system is not None
    tool = mapped.value.system.tools[0]
    assert tool.description == ""
    assert tool.json_schema == {"type": "object", "properties": {}}
