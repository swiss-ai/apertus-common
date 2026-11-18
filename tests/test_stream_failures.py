"""Structural failures are terminal and preserve a safe recovery boundary."""

import pytest

from apertus_common import (
    Conversation,
    MessageCompleted,
    ParseFailure,
    ParserStateError,
    Reply,
    StreamParseError,
    TextStreamParser,
    Wait,
    parse_output_text,
)


@pytest.mark.parametrize(
    "header",
    [
        "",
        " reply",
        "reply ",
        "reply{}",
        'reply{"x":1}',
        "type=reply",
        "call",
        "call{}",
        'call{"name":"x"}',
        'call{"name":"x","counter":true}',
        'call{"name":"x","counter":1.0}',
        'call{"name":"x","counter":"1"}',
        'call{"name":1,"counter":0}',
        'call{"name":"x","counter":0,"extra":1}',
        'call{"name":"x","counter":0,"counter":1}',
        'call{"name":"x y","counter":0}',
        'call{"name":"x\\ny","counter":0}',
        'call{"name":"x", "counter":0}',
        'call{"name":"x","counter":NaN}',
        'unknown{"x":1e999}',
        'unknown{"x":["bad\\tvalue"]}',
        'unknown{"x":"\\ud800"}',
        "unknown[]",
        "unknown{}garbage",
    ],
)
def test_bad_headers_fail_before_message_start(header):
    parser = TextStreamParser()
    with pytest.raises(StreamParseError) as caught:
        parser.push_text(f"<|out|>{header}<|hdr|>payload<|/out|>")
    assert caught.value.reason in {
        ParseFailure.INVALID_HEADER,
        ParseFailure.DUPLICATE_HEADER,
    }
    assert caught.value.offset == len("<|out|>")
    assert parser.take_events() == []


@pytest.mark.parametrize(
    "source,reason",
    [
        (" ", ParseFailure.UNEXPECTED_TEXT),
        ("<|hdr|>", ParseFailure.UNEXPECTED_CONTROL),
        ("<|in|>", ParseFailure.UNEXPECTED_CONTROL),
        ("<|pad|>", ParseFailure.UNEXPECTED_CONTROL),
        ("<|out|>reply<|/out|>", ParseFailure.MISMATCHED_CLOSE),
        ("<|out|>reply<|hdr|>a<|hdr|>", ParseFailure.UNEXPECTED_CONTROL),
        ("<|out|>reply<|hdr|>a<|wait|>", ParseFailure.UNEXPECTED_CONTROL),
        ("<|out|>reply<|hdr|>a<|in|>", ParseFailure.UNEXPECTED_CONTROL),
        ("<|out|>reply<|hdr|>a<|/in|>", ParseFailure.MISMATCHED_CLOSE),
        ("<|out|>reply<|hdr|>a<|/out|>\n", ParseFailure.UNEXPECTED_TEXT),
        ("<|wait|><|out|>", ParseFailure.TRAILING_AFTER_WAIT),
        ("<|wait|>x", ParseFailure.TRAILING_AFTER_WAIT),
        ("<|wait|><", ParseFailure.TRAILING_AFTER_WAIT),
    ],
)
def test_framing_errors(source, reason):
    with pytest.raises(StreamParseError) as caught:
        parse_output_text(source)
    assert caught.value.reason is reason
    assert caught.value.unit == "character"


@pytest.mark.parametrize(
    "source,reason",
    [
        ("<|ou", ParseFailure.INCOMPLETE_CONTROL),
        ("<|out|>", ParseFailure.INCOMPLETE_HEADER),
        ("<|out|>reply", ParseFailure.INCOMPLETE_HEADER),
        ("<|out|>reply<|hdr|>", ParseFailure.INCOMPLETE_PAYLOAD),
        ("<|out|>reply<|hdr|>hello", ParseFailure.INCOMPLETE_PAYLOAD),
    ],
)
def test_truncated_segments(source, reason):
    with pytest.raises(StreamParseError) as caught:
        parse_output_text(source)
    assert caught.value.reason is reason


def test_malformed_suffix_recovery_and_events():
    good = "<|out|>reply<|hdr|>done<|/out|>"
    parser = TextStreamParser()
    with pytest.raises(StreamParseError) as caught:
        parser.push_text(good + '<|out|>call{"counter":0}<|hdr|>bad<|/out|>')
    error = caught.value
    assert error.completed == Conversation(items=(Reply(payload="done"),))
    assert error.last_valid_boundary == len(good)
    assert error.index == 1
    assert (
        sum(isinstance(event, MessageCompleted) for event in parser.take_events()) == 1
    )
    with pytest.raises(ParserStateError):
        parser.push_text("anything")
    with pytest.raises(ParserStateError):
        parser.finish()


def test_wait_is_completed_on_trailing_failure():
    with pytest.raises(StreamParseError) as caught:
        parse_output_text("<|wait|>bad")
    assert caught.value.completed.items == (Wait(),)
    assert caught.value.last_valid_boundary == 8


def test_push_after_wait_or_finish_is_misuse():
    for source in ("", "<|wait|>", "<|out|>reply<|hdr|>x<|/out|>"):
        parser = TextStreamParser()
        parser.push_text(source)
        if source == "<|wait|>":
            with pytest.raises(ParserStateError):
                parser.push_text("")
        parser.finish()
        with pytest.raises(ParserStateError):
            parser.push_text("")


def test_excessive_header_nesting_is_a_structural_error():
    header = 'unknown{"x":' + "[" * 2000 + "0" + "]" * 2000 + "}"
    with pytest.raises(StreamParseError) as caught:
        parse_output_text(f"<|out|>{header}<|hdr|><|/out|>")
    assert caught.value.reason is ParseFailure.INVALID_HEADER
