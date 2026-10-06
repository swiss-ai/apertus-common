"""Canonical Apertus 2 rendering uses compact JSON and exact payloads."""

from apertus_common import (
    Call,
    Claim,
    Conversation,
    Document,
    Host,
    Reply,
    Result,
    SystemPrompt,
    Think,
    ToolSpec,
    UnknownOutput,
    User,
    Wait,
    render_str,
)


def test_compact_envelopes_preserve_payload_and_arrival_order():
    conversation = Conversation(
        items=[
            User(payload=" x \n"),
            Host(payload="y"),
            Call(name='a"b', counter=-1, payload="bad JSON"),
            Wait(),
        ]
    )
    assert (
        render_str(conversation)
        == '<|in|>user<|hdr|> x \n<|/in|><|in|>host<|hdr|>y<|/in|><|out|>call{"name":"a\\"b","counter":-1}<|hdr|>bad JSON<|/out|><|wait|>'
    )


def test_system_is_compact_json_in_regular_input_envelope():
    system = SystemPrompt.build(
        identity="A",
        tools=[ToolSpec(name="t", description="D", json_schema={"type": "object"})],
    )
    assert (
        render_str(Conversation(system=system))
        == '<|in|>system<|hdr|>{"identity":"A","thinking":"medium","tools":[{"name":"t","description":"D","schema":{"type":"object"}}]}<|/in|>'
    )


def test_system_without_tools_keeps_empty_tools_array():
    assert (
        render_str(Conversation(system=SystemPrompt.build()))
        == '<|in|>system<|hdr|>{"thinking":"medium","tools":[]}<|/in|>'
    )


def test_documents_and_empty_messages_add_no_payload_spaces():
    conversation = Conversation(
        items=[Document(payload="", from_="user", id="a"), Reply(payload="")]
    )
    assert (
        render_str(conversation)
        == '<|in|>document{"from":"user","id":"a"}<|hdr|><|/in|><|out|>reply<|hdr|><|/out|>'
    )


def test_result_think_claim_and_unknown_output_envelopes():
    conversation = Conversation(
        items=[
            Result(name="get_weather", counter=4, payload='{"tempC":24}'),
            Think(payload="Plan."),
            Claim(payload='{"x":3}'),
            UnknownOutput(type="quack", headers={"k": "v"}, payload="x"),
        ]
    )
    assert (
        render_str(conversation)
        == '<|in|>result{"name":"get_weather","counter":4}<|hdr|>{"tempC":24}<|/in|>'
        "<|out|>think<|hdr|>Plan.<|/out|>"
        '<|out|>claim<|hdr|>{"x":3}<|/out|>'
        '<|out|>quack{"k":"v"}<|hdr|>x<|/out|>'
    )
