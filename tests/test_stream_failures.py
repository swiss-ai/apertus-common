"""StreamParseError contract: reasons, positions, raw state, terminality."""

import json

import pytest
from pydantic import ValidationError

from apertus_common import (
    FramingError,
    HeaderError,
    IncompleteMessageError,
    MessageValidationError,
    ParseFailure,
    ParserStateError,
    StreamParseError,
    StreamParser,
    Think,
)

MESSAGE = "<|out|> type=think <|hdr|> plan <|/out|>"


def failing(text: str) -> StreamParseError:
    parser = StreamParser()
    with pytest.raises(StreamParseError) as info:
        parser.push(text)
        parser.finish()
    assert info.value.parser is parser
    return info.value


@pytest.mark.parametrize(
    ("text", "reason", "error_class", "state"),
    [
        ("junk", ParseFailure.UNEXPECTED_TEXT, FramingError, "expect_output"),
        ("<|sys|>", ParseFailure.UNEXPECTED_CONTROL, FramingError, "expect_output"),
        (
            MESSAGE + "\nx",
            ParseFailure.INVALID_SEPARATOR,
            FramingError,
            "separator",
        ),
        (
            "<|out|> type=assistant<|hdr|> x <|/out|>",
            ParseFailure.INVALID_HEADER,
            FramingError,
            "header",
        ),
        (
            "<|out|> type=a k=v k=w <|hdr|> x <|/out|>",
            ParseFailure.DUPLICATE_HEADER,
            FramingError,
            "header",
        ),
        (
            "<|out|> k=v <|hdr|> x <|/out|>",
            ParseFailure.MISSING_TYPE,
            FramingError,
            "header",
        ),
        (
            "<|out|> type=a <|hdr|> x<|/out|>",
            ParseFailure.MISMATCHED_CLOSE,
            FramingError,
            "payload",
        ),
        (
            MESSAGE + "\n<|wait|>trailing",
            ParseFailure.TRAILING_AFTER_WAIT,
            FramingError,
            "waiting",
        ),
        (
            "<|out",
            ParseFailure.INCOMPLETE_CONTROL,
            IncompleteMessageError,
            "expect_output",
        ),
        (
            "<|out|> type=a",
            ParseFailure.INCOMPLETE_HEADER,
            IncompleteMessageError,
            "header",
        ),
        (
            "<|out|> type=a <|hdr|> x",
            ParseFailure.INCOMPLETE_PAYLOAD,
            IncompleteMessageError,
            "payload",
        ),
        (
            MESSAGE + "\n",
            ParseFailure.INCOMPLETE_SEPARATOR,
            IncompleteMessageError,
            "separator",
        ),
        (
            "<|out|> type=tool_call id=a:0 <|hdr|> nope <|/out|>",
            ParseFailure.INVALID_TOOL_PAYLOAD,
            MessageValidationError,
            "separator",
        ),
        (
            (
                "<|out|> type=tool_call id=a:0 <|hdr|> "
                '{"name":"bad name","args":{}} <|/out|>'
            ),
            ParseFailure.MESSAGE_VALIDATION,
            MessageValidationError,
            "separator",
        ),
    ],
)
def test_reason_class_and_state(
    text: str, reason: ParseFailure, error_class: type[StreamParseError], state: str
) -> None:
    error = failing(text)
    assert isinstance(error, error_class)
    assert error.reason is reason
    assert error.state == state


def test_every_failure_reason_is_exercised_here() -> None:
    listed = {
        ParseFailure.UNEXPECTED_TEXT,
        ParseFailure.UNEXPECTED_CONTROL,
        ParseFailure.INVALID_SEPARATOR,
        ParseFailure.INVALID_HEADER,
        ParseFailure.DUPLICATE_HEADER,
        ParseFailure.MISSING_TYPE,
        ParseFailure.MISMATCHED_CLOSE,
        ParseFailure.INVALID_TOOL_PAYLOAD,
        ParseFailure.MESSAGE_VALIDATION,
        ParseFailure.TRAILING_AFTER_WAIT,
        ParseFailure.INCOMPLETE_CONTROL,
        ParseFailure.INCOMPLETE_HEADER,
        ParseFailure.INCOMPLETE_PAYLOAD,
        ParseFailure.INCOMPLETE_SEPARATOR,
    }
    assert listed == set(ParseFailure)


def test_output_scope_rejects_every_foreign_control() -> None:
    for glyph in (
        "<|sys|>",
        "<|/sys|>",
        "<|in|>",
        "<|/in|>",
        "<|pad|>",
    ):
        error = failing(glyph)
        assert error.reason is ParseFailure.UNEXPECTED_CONTROL, glyph


def test_offset_ordinal_and_unconsumed_suffix() -> None:
    text = MESSAGE + "\n\n<|out|> type=a <|hdr|> x<|/out|>"
    error = failing(text)
    assert error.reason is ParseFailure.MISMATCHED_CLOSE
    assert error.message_ordinal == 1
    assert error.offset == text.index("<|/out|>", len(MESSAGE))
    assert error.unconsumed == "<|/out|>"
    assert error.raw_text == text[len(MESSAGE) :]


def test_prior_messages_and_deltas_survive_failure() -> None:
    text = MESSAGE + "\n\njunk"
    parser = StreamParser()
    with pytest.raises(FramingError) as info:
        parser.push(text)
    assert info.value.parser is parser
    assert parser.messages == (Think(payload="plan"),)
    assert parser.take_completed() == [Think(payload="plan")]
    deltas = parser.take_deltas()
    assert "".join(delta.text for delta in deltas) == "plan"


def test_failed_message_deltas_are_preserved() -> None:
    parser = StreamParser()
    parser.push("<|out|> type=assistant <|hdr|> visible")
    assert parser.take_deltas() != []
    with pytest.raises(FramingError):
        parser.push(" <|sys|>")
    remaining = parser.take_deltas()
    assert remaining != []


def test_causes_are_chained() -> None:
    header = failing("<|out|> type= <|hdr|> x <|/out|>")
    assert isinstance(header.__cause__, HeaderError)
    tool_json = failing(
        '<|out|> type=tool_call id=a:0 <|hdr|> {"name":"a","args":NaN} <|/out|>'
    )
    assert isinstance(tool_json.__cause__, ValueError)
    bad_json = failing("<|out|> type=tool_call id=a:0 <|hdr|> {broken <|/out|>")
    assert isinstance(bad_json.__cause__, json.JSONDecodeError)
    model = failing(
        '<|out|> type=tool_call id=a:0 <|hdr|> {"name":"no way","args":{}} <|/out|>'
    )
    assert isinstance(model.__cause__, ValidationError)


def test_failure_is_permanent_and_reported_once() -> None:
    parser = StreamParser()
    with pytest.raises(FramingError):
        parser.push("junk")
    with pytest.raises(ParserStateError):
        parser.push("<|out|>")
    with pytest.raises(ParserStateError):
        parser.finish()
    assert parser.messages == ()


def test_batch_error_carries_the_terminal_batch_parser() -> None:
    with pytest.raises(FramingError) as info:
        StreamParser.parse(MESSAGE + "\n\njunk")
    parser = info.value.parser
    assert parser.messages == (Think(payload="plan"),)
    with pytest.raises(ParserStateError):
        parser.push("more")


def test_incomplete_control_reports_prefix_position() -> None:
    parser = StreamParser()
    parser.push(MESSAGE + "\n\n<|ou")
    with pytest.raises(IncompleteMessageError) as info:
        parser.finish()
    error = info.value
    assert error.reason is ParseFailure.INCOMPLETE_CONTROL
    assert error.unconsumed == "<|ou"
    assert error.raw_text.endswith("<|ou")


def test_message_validation_keeps_message_out_of_queues() -> None:
    parser = StreamParser()
    with pytest.raises(MessageValidationError):
        parser.push(
            '<|out|> type=tool_call id=a:0 <|hdr|> {"name":1,"args":{}} <|/out|>'
        )
    assert parser.messages == ()
    assert parser.take_completed() == []
