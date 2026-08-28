"""Source-neutral lexemes: the boundary between frontends and the machine.

A lexer frontend turns raw source (text now, token ids later) into a flat
stream of lexemes; the output machine consumes only lexemes and reports
positions through spans. Spans let the machine locate failures without
storing source-specific values in each lexeme.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from apertus_common.core._wire import ControlToken


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A half-open source range ``[start, end)`` in ``unit`` positions."""

    start: int
    end: int
    unit: Literal["character", "token"]


@dataclass(frozen=True, slots=True)
class TextLexeme:
    """A maximal run of ordinary source text."""

    text: str
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class ControlLexeme:
    """One recognized control token."""

    token: ControlToken
    span: SourceSpan


Lexeme = TextLexeme | ControlLexeme


class Lexer[InputT](Protocol):
    """The frontend protocol: push source units, flush at end of input."""

    def push(self, value: InputT) -> Sequence[Lexeme]: ...

    def finish(self) -> Sequence[Lexeme]: ...
