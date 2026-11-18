"""TextLexer: control-glyph scanning, spans, and prefix holdback.

Imports private core modules deliberately: the design's implementation
order tests the lexer before the public facade exists, and the lexeme
layer never becomes public API.
"""

import dataclasses
import random

import pytest

from apertus_common.core._lexeme import ControlLexeme, Lexeme, SourceSpan, TextLexeme
from apertus_common.core._scan import TextLexer
from apertus_common.core._wire import ControlToken

MIXED = "<|out|>reply<|hdr|>a <b> c<|/out|><|out|>think<|hdr|>d<|/out|><|wait|>"


def merged(lexemes: list[Lexeme]) -> list[tuple[str, str, int, int]]:
    """Normalize a lexeme stream: adjacent text runs merge into one record."""
    result: list[tuple[str, str, int, int]] = []
    for lexeme in lexemes:
        if isinstance(lexeme, ControlLexeme):
            result.append(
                ("control", lexeme.token.value, lexeme.span.start, lexeme.span.end)
            )
        elif result and result[-1][0] == "text" and result[-1][3] == lexeme.span.start:
            _, text, start, _ = result[-1]
            result[-1] = ("text", text + lexeme.text, start, lexeme.span.end)
        else:
            result.append(("text", lexeme.text, lexeme.span.start, lexeme.span.end))
    return result


def scan(chunks: list[str]) -> tuple[TextLexer, list[Lexeme]]:
    lexer = TextLexer()
    lexemes: list[Lexeme] = []
    for chunk in chunks:
        lexemes.extend(lexer.push(chunk))
    lexemes.extend(lexer.finish())
    return lexer, lexemes


def test_source_span_is_frozen_and_comparable() -> None:
    span = SourceSpan(0, 3, "character")
    assert span == SourceSpan(0, 3, "character")
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.start = 1  # type: ignore[misc]


def test_every_glyph_is_recognized() -> None:
    for token in ControlToken:
        lexer, lexemes = scan([token.value])
        assert lexemes == [
            ControlLexeme(token, SourceSpan(0, len(token.value), "character"))
        ]
        assert lexer.pending == ""


def test_maximal_text_runs_between_controls() -> None:
    _, lexemes = scan(["abc<|out|>def"])
    assert lexemes == [
        TextLexeme("abc", SourceSpan(0, 3, "character")),
        ControlLexeme(ControlToken.OUT, SourceSpan(3, 10, "character")),
        TextLexeme("def", SourceSpan(10, 13, "character")),
    ]


def test_glyph_split_across_two_chunks() -> None:
    _, lexemes = scan(["a<|ou", "t|>b"])
    assert merged(lexemes) == [
        ("text", "a", 0, 1),
        ("control", "<|out|>", 1, 8),
        ("text", "b", 8, 9),
    ]


def test_glyph_split_per_character() -> None:
    _, lexemes = scan(list("<|wait|>"))
    assert lexemes == [ControlLexeme(ControlToken.WAIT, SourceSpan(0, 8, "character"))]


def test_false_prefix_is_flushed_as_text() -> None:
    lexer = TextLexer()
    first = list(lexer.push("a<"))
    assert first == [TextLexeme("a", SourceSpan(0, 1, "character"))]
    assert lexer.pending == "<"
    second = list(lexer.push("b"))
    assert second == [TextLexeme("<b", SourceSpan(1, 3, "character"))]
    assert lexer.pending == ""


def test_plain_angle_bracket_text_is_not_delayed() -> None:
    _, lexemes = scan(["value < threshold > limit"])
    assert merged(lexemes) == [("text", "value < threshold > limit", 0, 25)]


def test_pending_after_truncated_glyph() -> None:
    lexer = TextLexer()
    assert list(lexer.push("<|wai")) == []
    assert lexer.pending == "<|wai"
    assert lexer.pending_start == 0
    assert list(lexer.finish()) == []
    assert lexer.pending == "<|wai"


def test_pending_offset_tracks_absolute_position() -> None:
    lexer = TextLexer()
    lexer.push("abc")
    lexer.push("<|out")
    assert lexer.pending == "<|out"
    assert lexer.pending_start == 3


def test_split_invariance_across_chunkings() -> None:
    whole = merged(scan([MIXED])[1])
    per_character = merged(scan(list(MIXED))[1])
    rng = random.Random(7)
    chunks: list[str] = []
    position = 0
    while position < len(MIXED):
        size = rng.randint(1, 9)
        chunks.append(MIXED[position : position + size])
        position += size
    random_chunks = merged(scan(chunks)[1])
    assert whole == per_character == random_chunks


def test_empty_push_is_a_no_op() -> None:
    lexer = TextLexer()
    assert list(lexer.push("")) == []
    assert lexer.pending == ""
