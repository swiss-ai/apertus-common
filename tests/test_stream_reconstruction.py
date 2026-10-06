"""Chunk boundaries cannot change messages, payloads or failure locations."""

import random

import pytest

from apertus_common import (
    MessageCompleted,
    PayloadDelta,
    StreamParseError,
    TextStreamParser,
    parse_output_text,
)

FIXTURES = [
    "<|out|>reply<|hdr|>  leading\ntrailing  <|/out|><|wait|>",
    '<|out|>call{"name":"tool","counter":-3}<|hdr|>not JSON<|/out|>',
    '<|out|>future{"nested":[{"k":"v"}]}<|hdr|>雪e\u0301🚀<|/out|>',
    "<|out|>think<|hdr|>x<|/out|><|out|>reply<|hdr|><|/out|>",
]


@pytest.mark.parametrize("source", FIXTURES)
def test_every_two_way_split(source):
    expected = parse_output_text(source)
    for split in range(len(source) + 1):
        parser = TextStreamParser()
        # Never push a second empty chunk after a terminal wait.
        for chunk in (source[:split], source[split:]):
            if chunk:
                parser.push_text(chunk)
        assert parser.finish() == expected
        events = parser.take_events()
        for index, item in enumerate(expected.items):
            if hasattr(item, "payload"):
                assert (
                    "".join(
                        e.text
                        for e in events
                        if isinstance(e, PayloadDelta) and e.index == index
                    )
                    == item.payload
                )
        assert [e.message for e in events if isinstance(e, MessageCompleted)] == [
            i for i in expected.items if hasattr(i, "payload")
        ]


def test_seeded_large_chunks_and_character_chunks():
    source = "".join(FIXTURES[1:] * 200) + "<|wait|>"
    expected = parse_output_text(source)
    for max_chunk in (1, 7, 193):
        randomizer = random.Random(123)
        parser = TextStreamParser()
        offset = 0
        while offset < len(source):
            size = randomizer.randint(1, max_chunk)
            parser.push_text(source[offset : offset + size])
            parser.take_events()
            offset += size
        assert parser.finish() == expected


def test_split_failure_location_is_stable():
    source = '<|out|>reply<|hdr|>ok<|/out|><|out|>call{"name":false,"counter":0}<|hdr|>'
    offsets = set()
    for split in range(len(source) + 1):
        parser = TextStreamParser()
        with pytest.raises(StreamParseError) as caught:
            parser.push_text(source[:split])
            parser.push_text(source[split:])
            parser.finish()
        offsets.add(
            (caught.value.reason, caught.value.offset, caught.value.last_valid_boundary)
        )
    assert len(offsets) == 1
