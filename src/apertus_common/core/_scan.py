"""Text frontend: a streaming lexer with control-prefix holdback.

The lexer recognizes the complete control-glyph inventory with one
precompiled longest-first literal alternation, emits maximal text runs and
individual control lexemes with absolute character spans, and holds back
only the longest buffer suffix that is a proper prefix of a glyph, so glyphs
may straddle arbitrary chunk boundaries without delaying unrelated text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from apertus_common.core._lexeme import ControlLexeme, Lexeme, SourceSpan, TextLexeme
from apertus_common.core._wire import ControlToken

if TYPE_CHECKING:
    from apertus_common.core._lexeme import Lexer

_GLYPHS = tuple(sorted((token.value for token in ControlToken), key=len, reverse=True))
_CONTROL_RE = re.compile("|".join(re.escape(glyph) for glyph in _GLYPHS))
_TOKEN_BY_GLYPH = {token.value: token for token in ControlToken}
# proper prefixes of every glyph; each contains "<" exactly once, at index 0
_PREFIXES = frozenset(
    glyph[:length] for glyph in _GLYPHS for length in range(1, len(glyph))
)


class TextLexer:
    """Incremental character-mode lexer over the control-glyph inventory.

    ``push`` accepts arbitrary chunks and returns completed lexemes;
    ``finish`` returns nothing in text mode because any residue is an
    unterminated glyph prefix, exposed through ``pending`` for the facade
    to report as an incomplete-control failure.
    """

    def __init__(self) -> None:
        self._held = ""
        self._offset = 0

    @property
    def pending(self) -> str:
        """The held glyph prefix awaiting more input; empty when clean."""
        return self._held

    @property
    def pending_start(self) -> int:
        """Absolute character offset of the first held character."""
        return self._offset

    def push(self, value: str) -> Sequence[Lexeme]:
        """Scan *value* and return the lexemes it completes."""
        buffer = self._held + value
        base = self._offset
        lexemes: list[Lexeme] = []
        position = 0
        for match in _CONTROL_RE.finditer(buffer):
            if match.start() > position:
                lexemes.append(
                    TextLexeme(
                        buffer[position : match.start()],
                        SourceSpan(base + position, base + match.start(), "character"),
                    )
                )
            lexemes.append(
                ControlLexeme(
                    _TOKEN_BY_GLYPH[match.group()],
                    SourceSpan(base + match.start(), base + match.end(), "character"),
                )
            )
            position = match.end()
        tail = buffer[position:]
        # a glyph prefix contains "<" only at index 0, so the only suffix
        # that can continue a glyph starts at the last "<" in the tail
        cut = len(tail)
        marker = tail.rfind("<")
        if marker >= 0 and tail[marker:] in _PREFIXES:
            cut = marker
        if cut:
            lexemes.append(
                TextLexeme(
                    tail[:cut],
                    SourceSpan(base + position, base + position + cut, "character"),
                )
            )
        self._held = tail[cut:]
        self._offset = base + position + cut
        return lexemes

    def finish(self) -> Sequence[Lexeme]:
        """End of input; text mode never completes lexemes here."""
        return ()


if TYPE_CHECKING:
    # static conformance check: TextLexer implements the frontend protocol
    _TEXT_LEXER_IS_A_LEXER: Lexer[str] = TextLexer()
