"""Open output messages: prefill rendering and truncated-output parsing."""

import json

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from apertus_common import (
    AdapterError,
    Call,
    Claim,
    Conversation,
    Document,
    Host,
    MappingPolicy,
    Message,
    MessageCompleted,
    MessageStart,
    OpenAIStreamMapper,
    ParseFailure,
    PayloadDelta,
    Reply,
    Result,
    StreamParseError,
    SystemPrompt,
    TextStreamParser,
    Think,
    UnknownOutput,
    User,
    Wait,
    parse_output_text,
    render_str,
    to_openai,
    to_openai_output,
)

OPEN_ITEMS = [
    Think(payload="partial", open=True),
    Reply(payload="", open=True),
    Claim(payload="4", open=True),
    Call(name="lookup", counter=0, payload='{"q":', open=True),
    UnknownOutput(output_type="future", headers={"x": "y"}, payload="z", open=True),
]


def test_closed_by_default_with_unchanged_repr_and_json():
    reply = Reply(payload="x")
    assert reply.open is False
    assert reply == Reply(payload="x", open=False) != Reply(payload="x", open=True)
    assert "open" not in repr(reply)
    assert "open=True" in repr(Reply(payload="x", open=True))
    assert reply.to_json() == '{"type":"reply","payload":"x"}'


@pytest.mark.parametrize("message", [User, Host, Document, Result])
def test_only_output_messages_can_be_open(message):
    fields = {"name": "t", "counter": 0} if message is Result else {}
    with pytest.raises(ValidationError):
        message(payload="x", open=True, **fields)


def test_open_item_must_be_last():
    for tail in [Wait(), User(payload="next"), Reply(payload="closed")]:
        with pytest.raises(ValidationError, match="only the last item"):
            Conversation(items=[Reply(payload="a", open=True), tail])
    assert Conversation(items=[User(payload="q"), Reply(payload="a", open=True)])


@pytest.mark.parametrize(
    "item",
    [
        Reply(payload="", open=True),
        UnknownOutput(output_type="f", payload="z", open=True),
    ],
)
def test_native_json_rejects_open_messages(item):
    with pytest.raises(ValidationError, match="no native JSON"):
        item.to_json()
    with pytest.raises(ValidationError, match="no native JSON"):
        Conversation(items=[item]).to_json()
    data = json.loads(item.model_copy(update={"open": False}).to_json())
    assert "open" not in data
    for value in [True, False]:
        with pytest.raises(ValidationError, match="no open messages"):
            type(item).from_json(json.dumps({**data, "open": value}))
        document = {"schema_version": 2, "items": [{**data, "open": value}]}
        with pytest.raises(ValidationError, match="no open messages"):
            Conversation.from_json(json.dumps(document))


def test_open_messages_stay_in_memory():
    item = Reply(payload="a", open=True)
    conversation = Conversation(items=[User(payload="q"), item])
    for dump in [
        item.model_dump_json,
        lambda: item.model_dump(mode="json"),
        conversation.model_dump_json,
    ]:
        with pytest.raises(PydanticSerializationError, match="no JSON"):
            dump()
    assert item.model_dump() == {"type": "reply", "payload": "a"}
    with pytest.raises(PydanticSerializationError, match="no JSON"):
        TypeAdapter(Message).dump_json(item)
    data = '{"schema_version":2,"items":[{"type":"reply","payload":"x","open":true}]}'
    with pytest.raises(ValidationError, match="no open messages"):
        Conversation.model_validate_json(data)
    for cls in (Think, Reply, Claim, Call, UnknownOutput):
        assert "payload" in cls.model_json_schema(mode="serialization")["required"]
    copied = conversation.model_copy(update={"items": [item, Wait()]})
    with pytest.raises(ValidationError, match="no native JSON"):
        copied.to_json()


@pytest.mark.parametrize(
    "item,text",
    [
        (Think(payload="t", open=True), "<|out|>think<|hdr|>t"),
        (Reply(payload="", open=True), "<|out|>reply<|hdr|>"),
        (
            Call(name="f", counter=1, payload="{", open=True),
            '<|out|>call{"name":"f","counter":1}<|hdr|>{',
        ),
        (
            UnknownOutput(output_type="f", headers={"x": "y"}, payload="z", open=True),
            '<|out|>f{"x":"y"}<|hdr|>z',
        ),
    ],
)
def test_open_items_render_without_close(item, text):
    assert render_str(Conversation(items=[item])) == text


@pytest.mark.parametrize(
    "source,items",
    [
        ("<|out|>reply<|hdr|>", (Reply(payload="", open=True),)),
        (
            "<|out|>think<|hdr|>t<|/out|><|out|>reply<|hdr|>par",
            (Think(payload="t"), Reply(payload="par", open=True)),
        ),
        (
            "<|out|>think<|hdr|>still thinking",
            (Think(payload="still thinking", open=True),),
        ),
        (
            '<|out|>call{"counter":0,"name":"lookup"}<|hdr|>{"q":',
            (Call(name="lookup", counter=0, payload='{"q":', open=True),),
        ),
        (
            '<|out|>future{"x":"y"}<|hdr|>z',
            (
                UnknownOutput(
                    output_type="future", headers={"x": "y"}, payload="z", open=True
                ),
            ),
        ),
        ("<|out|>reply<|hdr|>a <", (Reply(payload="a <", open=True),)),
        ("<|out|>reply<|hdr|>a <|/ou", (Reply(payload="a <|/ou", open=True),)),
        ("<|out|>reply<|hdr|>a<|/out|><|wait|>", (Reply(payload="a"), Wait())),
        ("", ()),
    ],
)
def test_text_parser_allow_open(source, items):
    assert parse_output_text(source, allow_open=True) == Conversation(items=items)
    for size in range(1, len(source) + 1):
        parser = TextStreamParser()
        for offset in range(0, len(source), size):
            parser.push_text(source[offset : offset + size])
        assert parser.finish(allow_open=True) == Conversation(items=items)


def test_open_payload_without_allow_open_still_fails():
    with pytest.raises(StreamParseError) as caught:
        parse_output_text("<|out|>think<|hdr|>t<|/out|><|out|>reply<|hdr|>par")
    assert caught.value.reason is ParseFailure.INCOMPLETE_PAYLOAD
    assert caught.value.completed == Conversation(items=[Think(payload="t")])


@pytest.mark.parametrize(
    "source,reason",
    [
        ("<|out|>reply", ParseFailure.INCOMPLETE_HEADER),
        ("<|out|>reply<|hd", ParseFailure.INCOMPLETE_CONTROL),
        ("<|out|>reply<|hdr|>a<|/out|><|ou", ParseFailure.INCOMPLETE_CONTROL),
        ("<|out|>reply<|hdr|>a\ud800", ParseFailure.MESSAGE_VALIDATION),
    ],
)
def test_allow_open_still_requires_complete_header_and_controls(source, reason):
    with pytest.raises(StreamParseError) as caught:
        parse_output_text(source, allow_open=True)
    assert caught.value.reason is reason


def test_open_message_emits_no_completion_and_finish_is_idempotent():
    parser = TextStreamParser()
    parser.push_text("<|out|>reply<|hdr|>a <")
    result = parser.finish(allow_open=True)
    events = parser.take_events()
    assert not any(isinstance(event, MessageCompleted) for event in events)
    assert "".join(e.text for e in events if isinstance(e, PayloadDelta)) == "a <"
    assert parser.finish() is result
    assert parser.last_valid_boundary == 0


@pytest.mark.parametrize("think", ["reasoning", ""])
def test_prefill_renders_and_its_continuation_parses(think):
    prefix = "The best answer is"
    prompt = Conversation(
        system=SystemPrompt.build(),
        items=[
            User(payload="Q?"),
            Think(payload=think),
            Reply(payload=prefix, open=True),
        ],
    )
    head = render_str(
        Conversation(system=SystemPrompt.build(), items=[User(payload="Q?")])
    )
    assert render_str(prompt) == (
        f"{head}<|out|>think<|hdr|>{think}<|/out|><|out|>reply<|hdr|>{prefix}"
    )
    prefill = Conversation(items=prompt.items[1:])
    expected = Conversation(
        items=[Think(payload=think), Reply(payload=f"{prefix} B."), Wait()]
    )
    output = render_str(prefill) + " B.<|/out|><|wait|>"
    assert parse_output_text(output) == expected
    truncated = parse_output_text(render_str(prefill) + " B", allow_open=True)
    assert truncated.items[-1] == Reply(payload=f"{prefix} B", open=True)


@pytest.mark.parametrize("item", OPEN_ITEMS)
def test_mapping_rejects_open_messages(item):
    with pytest.raises(AdapterError, match="open messages") as caught:
        to_openai_output(Conversation(items=[Think(payload="t"), item]))
    assert caught.value.index == 1
    with pytest.raises(AdapterError, match="open messages") as caught:
        to_openai(Conversation(items=[User(payload="q"), item]))
    assert caught.value.index == 1


def test_stream_mapper_rejects_open_messages():
    mapper = OpenAIStreamMapper(None, MappingPolicy())
    mapper.push(MessageStart(0, "reply", {}))
    mapper.push(PayloadDelta(0, "a"))
    with pytest.raises(AdapterError, match="open messages"):
        mapper.push(MessageCompleted(0, Reply(payload="a", open=True)))


@pytest.mark.parametrize("mode", ["profile", "conversation", "full"])
def test_checkers_report_open_messages(mode):
    conversation = Conversation(
        system=SystemPrompt.build(),
        items=[User(payload="q"), Reply(payload="a", open=True)],
    )
    rules = {violation.rule for violation in conversation.check(mode).errors}
    assert "profile/closed-output" in rules
    closed = Conversation(
        system=SystemPrompt.build(), items=[User(payload="q"), Reply(payload="a")]
    )
    assert "profile/closed-output" not in {
        violation.rule for violation in closed.check(mode).errors
    }
