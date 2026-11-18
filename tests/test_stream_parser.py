"""Batch and streaming parsing reconstruct native values without conformance."""

import pytest

from apertus_common import (
    Call,
    Claim,
    Conversation,
    MessageCompleted,
    MessageStart,
    OpenMessage,
    ParseBoundary,
    PayloadDelta,
    Reply,
    TextStreamParser,
    Think,
    UnknownOutput,
    Wait,
    WaitReached,
    parse_output_text,
)


@pytest.mark.parametrize(
    "source,items",
    [
        ("", ()),
        ("<|wait|>", (Wait(),)),
        ("<|out|>reply<|hdr|><|/out|>", (Reply(payload=""),)),
        (
            "<|out|>think<|hdr|> x \n<|/out|><|out|>claim<|hdr|>42<|/out|><|wait|>",
            (Think(payload=" x \n"), Claim(payload="42"), Wait()),
        ),
        (
            '<|out|>call{"counter":-7,"name":"missing"}<|hdr|>not JSON<|/out|>',
            (Call(name="missing", counter=-7, payload="not JSON"),),
        ),
        (
            '<|out|>future{"nested":[{"x":"y"}]}<|hdr|>\u0308🚀<|/out|>',
            (
                UnknownOutput(
                    output_type="future",
                    headers={"nested": [{"x": "y"}]},
                    payload="\u0308🚀",
                ),
            ),
        ),
    ],
)
def test_batch_native_values(source, items):
    assert parse_output_text(source) == Conversation(items=items)


def test_ordered_events_and_current_snapshot():
    parser = TextStreamParser()
    parser.push_text("<|out|>reply<|hdr|>hi")
    assert parser.current == OpenMessage(0, "reply", {}, "hi")
    assert parser.boundary is ParseBoundary.IN_MESSAGE
    assert parser.take_events() == [MessageStart(0, "reply", {}), PayloadDelta(0, "hi")]
    parser.push_text("!<|/out|><|wait|>")
    assert parser.current is None
    result = parser.finish()
    events = parser.take_events()
    assert events == [
        PayloadDelta(0, "!"),
        MessageCompleted(0, Reply(payload="hi!")),
        WaitReached(),
    ]
    assert events[1].message == result.items[0]
    assert parser.take_events() == []
    assert parser.finish() is result
    assert parser.boundary is ParseBoundary.WAIT
    assert parser.consumed == len("<|out|>reply<|hdr|>hi!<|/out|><|wait|>")


def test_parser_does_not_run_checks_or_argument_validation(monkeypatch):
    from apertus_common import checker
    from apertus_common.model import Message

    def fail(*args, **kwargs):
        raise AssertionError("parser invoked a checker")

    monkeypatch.setattr(Conversation, "check", fail)
    monkeypatch.setattr(Message, "check", fail)
    monkeypatch.setattr(checker, "parse_arguments", fail)
    monkeypatch.setattr(checker, "check_arguments", fail)
    result = parse_output_text(
        '<|out|>call{"name":"unknown","counter":-1}<|hdr|>[invalid<|/out|>'
    )
    assert result.items == (Call(name="unknown", counter=-1, payload="[invalid"),)


def test_empty_and_clean_end_boundaries():
    parser = TextStreamParser()
    assert parser.boundary is ParseBoundary.EMPTY
    parser.push_text("")
    assert parser.finish() == Conversation()
    parser = TextStreamParser()
    parser.push_text("<|out|>reply<|hdr|>done<|/out|>")
    assert parser.boundary is ParseBoundary.MESSAGE_END
    assert parser.finish().items == (Reply(payload="done"),)
