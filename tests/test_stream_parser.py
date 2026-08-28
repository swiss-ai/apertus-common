"""StreamParser: streaming behavior, queues, tool JSON, and lifecycle."""

import pytest

from apertus_common import (
    Assistant,
    GenericOutput,
    MessageValidationError,
    ParseFailure,
    ParserStateError,
    PayloadDelta,
    StreamParseError,
    StreamParser,
    Think,
    ToolCall,
)

TWO_MESSAGES = (
    "<|out|> type=think <|hdr|> plan <|/out|>\n\n"
    "<|out|> type=assistant <|hdr|> hello <|/out|>"
)


def test_uneven_chunks_produce_messages_and_deltas() -> None:
    parser = StreamParser()
    deltas: list[PayloadDelta] = []
    completed = []
    for chunk in ["<|out|> ty", "pe=think <|h", "dr|> pl", "an <|/o", "ut|>"]:
        parser.push(chunk)
        deltas.extend(parser.take_deltas())
        completed.extend(parser.take_completed())
    parser.finish()
    assert completed == [Think(payload="plan")]
    assert "".join(delta.text for delta in deltas) == "plan"
    assert all(delta.ordinal == 0 and delta.type == "think" for delta in deltas)


def test_one_chunk_crossing_message_types_keeps_delta_identity() -> None:
    parser = StreamParser()
    parser.push(TWO_MESSAGES)
    parser.finish()
    deltas = parser.take_deltas()
    by_ordinal: dict[int, str] = {}
    types: dict[int, str] = {}
    for delta in deltas:
        by_ordinal[delta.ordinal] = by_ordinal.get(delta.ordinal, "") + delta.text
        types[delta.ordinal] = delta.type
    assert by_ordinal == {0: "plan", 1: "hello"}
    assert types == {0: "think", 1: "assistant"}
    assert parser.messages == (Think(payload="plan"), Assistant(payload="hello"))


def test_current_views_track_the_open_message() -> None:
    parser = StreamParser()
    assert parser.current_type is None
    assert dict(parser.current_headers) == {}
    assert parser.current_payload is None
    parser.push("<|out|> type=memo k=v <|hdr|> par")
    assert parser.current_type == "memo"
    assert dict(parser.current_headers) == {"k": "v"}
    assert parser.current_payload == "pa"
    parser.push("t <|/out|>")
    assert parser.current_type is None
    assert parser.current_payload is None
    parser.finish()
    assert parser.messages == (
        GenericOutput(type="memo", payload="part", extras={"k": "v"}),
    )


def test_queues_drain_while_messages_accumulate() -> None:
    parser = StreamParser()
    parser.push(TWO_MESSAGES)
    parser.finish()
    assert len(parser.take_completed()) == 2
    assert parser.take_completed() == []
    assert parser.take_deltas() != []
    assert parser.take_deltas() == []
    assert len(parser.messages) == 2


def test_wait_lifecycle() -> None:
    bare = StreamParser.parse("<|wait|>")
    assert bare.waiting
    assert bare.messages == ()
    attached = StreamParser.parse(TWO_MESSAGES + "\n<|wait|>")
    assert attached.waiting
    assert len(attached.messages) == 2
    unterminated = StreamParser.parse(TWO_MESSAGES)
    assert not unterminated.waiting


def test_two_space_body_is_not_an_empty_payload() -> None:
    text = "<|out|> type=assistant <|hdr|>  <|/out|>"
    for chunks in ([text], list(text)):
        parser = StreamParser()
        with pytest.raises(StreamParseError) as info:
            for chunk in chunks:
                parser.push(chunk)
            parser.finish()
        assert info.value.reason is ParseFailure.MISMATCHED_CLOSE


def test_empty_segment_finishes_clean() -> None:
    parser = StreamParser()
    parser.push("")
    parser.finish()
    assert parser.messages == ()
    assert not parser.waiting


def test_terminal_parser_rejects_further_input() -> None:
    parser = StreamParser.parse("<|wait|>")
    with pytest.raises(ParserStateError):
        parser.push("x")
    with pytest.raises(ParserStateError):
        parser.finish()
    failed = StreamParser()
    with pytest.raises(StreamParseError):
        failed.push("junk")
    with pytest.raises(ParserStateError):
        failed.push("<|out|>")
    with pytest.raises(ParserStateError):
        failed.finish()


def test_queries_stay_usable_on_terminal_parsers() -> None:
    parser = StreamParser.parse(TWO_MESSAGES)
    assert len(parser.messages) == 2
    assert parser.take_completed() != []
    assert parser.current_type is None


def test_tool_call_json_accepts_semantic_variants() -> None:
    reordered = (
        "<|out|> type=tool_call id=web:0 <|hdr|> "
        '{"args": {"q": 1}, "name": "web"} <|/out|>'
    )
    parser = StreamParser.parse(reordered)
    call = parser.messages[0]
    assert isinstance(call, ToolCall)
    assert call.name == "web"
    assert call.args == {"q": 1}
    spaced = (
        "<|out|> type=tool_call id=web:1 <|hdr|> "
        '{ "name" : "web" ,\n  "args" : [1, 2] } <|/out|>'
    )
    call = StreamParser.parse(spaced).messages[0]
    assert isinstance(call, ToolCall)
    assert call.args == [1, 2]


def test_tool_call_header_extras_survive() -> None:
    text = (
        "<|out|> type=tool_call id=web:0 note=x <|hdr|> "
        '{"name":"web","args":{}} <|/out|>'
    )
    call = StreamParser.parse(text).messages[0]
    assert isinstance(call, ToolCall)
    assert dict(call.extras) == {"note": "x"}


@pytest.mark.parametrize(
    "payload",
    [
        '{"name":"web"}',
        '{"name":"web","args":{},"more":1}',
        '["name","args"]',
        '"text"',
        '{"name":"web","args":{"a":1,"a":2}}',
        '{"name":"web","args":{"n":{"b":1,"b":2}}}',
        '{"name":"web","args":Infinity}',
        '{"name":"web","args":-Infinity}',
        '{"name":"web","args":NaN}',
        "not json",
    ],
)
def test_tool_call_json_rejections(payload: str) -> None:
    text = f"<|out|> type=tool_call id=web:0 <|hdr|> {payload} <|/out|>"
    parser = StreamParser()
    with pytest.raises(MessageValidationError) as info:
        parser.push(text)
        parser.finish()
    assert info.value.reason is ParseFailure.INVALID_TOOL_PAYLOAD
    assert parser.messages == ()
    assert parser.take_completed() == []


def test_tool_call_model_failures_are_message_validation() -> None:
    bad_name = (
        '<|out|> type=tool_call id=web:0 <|hdr|> {"name":"bad name","args":{}} <|/out|>'
    )
    with pytest.raises(MessageValidationError) as info:
        StreamParser.parse(bad_name)
    assert info.value.reason is ParseFailure.MESSAGE_VALIDATION
    bad_id = (
        '<|out|> type=tool_call id=bad^id <|hdr|> {"name":"web","args":{}} <|/out|>'
    )
    with pytest.raises(MessageValidationError) as info:
        StreamParser.parse(bad_id)
    assert info.value.reason is ParseFailure.MESSAGE_VALIDATION


def test_unknown_type_builds_generic_output_with_extras_in_order() -> None:
    text = "<|out|> type=custom b=2 a=1 <|hdr|> body <|/out|>"
    message = StreamParser.parse(text).messages[0]
    assert isinstance(message, GenericOutput)
    assert message.type == "custom"
    assert list(message.extras) == ["b", "a"]


def test_deltas_are_provisional_until_close() -> None:
    parser = StreamParser()
    parser.push("<|out|> type=assistant <|hdr|> stream")
    provisional = parser.take_deltas()
    assert "".join(delta.text for delta in provisional) == "strea"
    assert parser.take_completed() == []
    parser.push("ing <|/out|>")
    parser.finish()
    assert parser.messages == (Assistant(payload="streaming"),)
