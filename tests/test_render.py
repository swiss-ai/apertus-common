"""Canonical Apertus 2 rendering uses compact JSON and exact payloads."""

from apertus_common import (
    Call,
    Conversation,
    Document,
    Host,
    Reply,
    SystemPrompt,
    ToolSpec,
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


def test_documents_and_empty_messages_add_no_payload_spaces():
    conversation = Conversation(
        items=[Document(payload="", from_="user", id="a"), Reply(payload="")]
    )
    assert (
        render_str(conversation)
        == '<|in|>document{"from":"user","id":"a"}<|hdr|><|/in|><|out|>reply<|hdr|><|/out|>'
    )
