"""The wire IR: the middle layer between the typed model and the wire.

A conversation renders to a flat sequence of wire atoms. String and token
encoding project the same sequence, so framing cannot drift between them.
``Control`` atoms carry framing,
``Text`` atoms carry ordinary text with its trust origin. Framing is decided
in exactly one place, the renderer that produces the atoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class ControlToken(StrEnum):
    """The control-token inventory, name to surface glyph.

    Glyphs are the library's explicit map for the string projection; token
    encoding maps the same names to registered IDs from the tokenizer artifact.
    """

    IN = "<|in|>"
    IN_END = "<|/in|>"
    OUT = "<|out|>"
    OUT_END = "<|/out|>"
    HDR = "<|hdr|>"
    WAIT = "<|wait|>"
    PAD = "<|pad|>"


@dataclass(frozen=True, slots=True)
class Control:
    """One control token; projects to its glyph or its registered id."""

    token: ControlToken
    loss_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class Text:
    """A run of ordinary text.

    ``trust`` records the origin: ``structural`` text is renderer-written
    syntax while ``payload`` text carries message content. Both are encoded with
    special-token recognition disabled. ``loss_weight`` is training metadata
    and never enters the wire text.
    """

    text: str
    trust: Literal["structural", "payload"]
    loss_weight: float = 0.0


WireAtom = Control | Text
