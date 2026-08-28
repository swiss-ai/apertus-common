"""The wire IR: the middle layer between the typed model and the wire.

A conversation renders to a flat sequence of wire atoms. The string form is a
pure projection of that sequence, and the planned token form will be another,
so framing cannot drift between them. ``Control`` atoms carry framing,
``Text`` atoms carry ordinary text with its trust origin, and ``Media``
atoms carry inline placeholders. Framing is decided in exactly one place,
the renderer that produces the atoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class ControlToken(StrEnum):
    """The control-token inventory, name to surface glyph.

    Glyphs are the library's explicit map for the string projection; the
    planned token projection will map the same names to registered single token
    IDs from the tokenizer artifact.
    """

    SYS = "<|sys|>"
    SYS_END = "<|/sys|>"
    IN = "<|in|>"
    IN_END = "<|/in|>"
    OUT = "<|out|>"
    OUT_END = "<|/out|>"
    HDR = "<|hdr|>"
    WAIT = "<|wait|>"
    PAD = "<|pad|>"
    IMAGE = "<|image|>"
    AUDIO = "<|audio|>"
    IMG_START = "<|img_start|>"
    IMG_TOKEN_START = "<|img_token_start|>"
    IMG_END_OF_ROW = "<|img_end_of_row|>"
    IMG_END = "<|img_end|>"
    AUDIO_START = "<|audio_start|>"
    AUDIO_END = "<|audio_end|>"


@dataclass(frozen=True, slots=True)
class Control:
    """One control token; projects to its glyph or its registered id."""

    token: ControlToken


@dataclass(frozen=True, slots=True)
class Text:
    """A run of ordinary text.

    ``trust`` records the origin: ``structural`` text is renderer-written
    syntax (framing whitespace, header keys, tag markup) while ``payload``
    text carries model-derived content and must be encoded with
    special-token recognition disabled in the token projection.
    """

    text: str
    trust: Literal["structural", "payload"]


@dataclass(frozen=True, slots=True)
class Media:
    """An inline media placeholder; expansion belongs to the processor."""

    part: Literal["image", "audio"]


WireAtom = Control | Text | Media
