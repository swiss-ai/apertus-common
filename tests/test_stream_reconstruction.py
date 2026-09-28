"""Reconstruction: parsing rendered output reproduces the typed messages."""

import random

import pytest

from apertus_common import (
    Assistant,
    Conversation,
    GenericOutput,
    OutputMessage,
    PayloadDelta,
    StreamParseError,
    StreamParser,
    Think,
    ToolCall,
    VerifiableAnswer,
    Wait,
    render_str,
)

OUTPUT_RUN = Conversation(
    items=[
        Think(payload="weigh the options"),
        ToolCall(id="search:0", name="search", args={"q": "x", "n": 3}),
        ToolCall(id="search:1", name="search", args=[1, None, True]),
        Assistant(payload=" leading and trailing "),
        GenericOutput(type="memo", payload="", extras={"k": "v"}),
        VerifiableAnswer(payload="42"),
        Wait(),
    ]
)


def outputs(conversation: Conversation) -> tuple[OutputMessage, ...]:
    return tuple(item for item in conversation.items if isinstance(item, OutputMessage))


def chunked(text: str, seed: int) -> list[str]:
    rng = random.Random(seed)
    chunks: list[str] = []
    position = 0
    while position < len(text):
        size = rng.randint(1, 7)
        chunks.append(text[position : position + size])
        position += size
    return chunks


def feed(chunks: list[str]) -> tuple[StreamParser, list[PayloadDelta]]:
    parser = StreamParser()
    deltas: list[PayloadDelta] = []
    for chunk in chunks:
        parser.push(chunk)
        deltas.extend(parser.take_deltas())
    parser.finish()
    deltas.extend(parser.take_deltas())
    return parser, deltas


def test_rendered_output_run_round_trips() -> None:
    text = render_str(OUTPUT_RUN)
    parser = StreamParser.parse(text)
    assert parser.messages == outputs(OUTPUT_RUN)
    assert parser.waiting


def test_each_output_class_round_trips_alone() -> None:
    for message in outputs(OUTPUT_RUN):
        text = render_str(Conversation(items=[message]))
        parser = StreamParser.parse(text)
        assert parser.messages == (message,)
        assert not parser.waiting


def test_awkward_payloads_round_trip() -> None:
    payloads = ["", " ", "  ", "a\nb", "a\n\nb", " x ", "\tx\t", "e=mc2 <ok>"]
    for payload in payloads:
        message = Assistant(payload=payload)
        text = render_str(Conversation(items=[message]))
        parser = StreamParser.parse(text)
        assert parser.messages == (message,), repr(payload)


@pytest.mark.parametrize(
    "glyph",
    [
        "<|image|>",
        "<|audio|>",
        "<|img_start|>",
        "<|img_token_start|>",
        "<|img_end_of_row|>",
        "<|img_end|>",
        "<|audio_start|>",
        "<|audio_end|>",
    ],
)
def test_former_media_glyphs_are_literal_text(glyph: str) -> None:
    message = Assistant(payload=f"literal {glyph} tail")
    text = render_str(Conversation(items=[message]))
    for chunks in ([text], list(text)):
        parser, deltas = feed(chunks)
        assert parser.messages == (message,)
        assert parser.take_completed() == [message]
        assert "".join(delta.text for delta in deltas) == message.payload


def test_split_invariance_of_messages_and_deltas() -> None:
    text = render_str(OUTPUT_RUN)
    whole_parser, whole_deltas = feed([text])
    char_parser, char_deltas = feed(list(text))
    random_parser, random_deltas = feed(chunked(text, seed=11))

    assert whole_parser.messages == char_parser.messages == random_parser.messages

    def joined(deltas: list[PayloadDelta]) -> dict[int, str]:
        result: dict[int, str] = {}
        for delta in deltas:
            result[delta.ordinal] = result.get(delta.ordinal, "") + delta.text
        return result

    assert joined(whole_deltas) == joined(char_deltas) == joined(random_deltas)


def test_split_invariance_of_failures() -> None:
    text = render_str(OUTPUT_RUN).replace("search:1", "bad id", 1)

    def outcome(chunks: list[str]) -> tuple[object, ...]:
        try:
            parser, _ = feed(chunks)
        except StreamParseError as error:
            return (
                type(error),
                error.reason,
                error.offset,
                error.message_ordinal,
                error.state,
                len(error.parser.messages),
            )
        return (None, len(parser.messages))

    assert outcome([text]) == outcome(list(text)) == outcome(chunked(text, seed=3))


def test_batch_equals_streaming() -> None:
    text = render_str(OUTPUT_RUN)
    batch = StreamParser.parse(text)
    streaming, _ = feed(chunked(text, seed=5))
    assert batch.messages == streaming.messages
    assert batch.waiting == streaming.waiting


def test_concatenated_deltas_equal_final_payloads() -> None:
    text = render_str(OUTPUT_RUN)
    parser, deltas = feed(chunked(text, seed=13))
    per_message: dict[int, str] = {}
    for delta in deltas:
        per_message[delta.ordinal] = per_message.get(delta.ordinal, "") + delta.text
    for ordinal, message in enumerate(parser.messages):
        expected = per_message.get(ordinal, "")
        if isinstance(message, ToolCall):
            rebuilt = StreamParser.parse(
                f"<|out|> type=tool_call id={message.id} <|hdr|> {expected} <|/out|>"
            ).messages[0]
            assert rebuilt == message
        else:
            payload = getattr(message, "payload")  # noqa: B009
            assert expected == payload
