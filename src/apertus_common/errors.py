"""Typed failures for construction, mapping, encoding and output parsing."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from apertus_common.model import Conversation


class ApertusFormatError(Exception):
    """Base class for invalid Apertus format data."""


class ArgumentError(ApertusFormatError, ValueError):
    """Call arguments are invalid JSON or do not have an object root."""


class TokenizerError(ApertusFormatError):
    """An artifact, binding, or ordinary-text encoding is invalid."""


class AdapterError(ApertusFormatError):
    """An external-format mapping is unsupported or conflicting."""

    def __init__(
        self, where: Literal["message", "item", "system"], index: int, reason: str
    ) -> None:
        super().__init__(where, index, reason)
        self.where = where
        self.index = index
        self.reason = reason

    def __str__(self) -> str:
        if self.where == "system":
            return f"system prompt: {self.reason}"
        noun = "openai message" if self.where == "message" else "conversation item"
        return f"{noun} {self.index}: {self.reason}"


class ParseFailure(StrEnum):
    """Stable structural failure reasons; argument validation is separate."""

    UNEXPECTED_TEXT = "unexpected_text"
    UNEXPECTED_CONTROL = "unexpected_control"
    INVALID_HEADER = "invalid_header"
    DUPLICATE_HEADER = "duplicate_header"
    MISMATCHED_CLOSE = "mismatched_close"
    MESSAGE_VALIDATION = "message_validation"
    TRAILING_AFTER_WAIT = "trailing_after_wait"
    INCOMPLETE_CONTROL = "incomplete_control"
    INCOMPLETE_HEADER = "incomplete_header"
    INCOMPLETE_PAYLOAD = "incomplete_payload"
    INVALID_ENCODING = "invalid_encoding"


class StreamParseError(ApertusFormatError):
    """Terminal structural failure with a safe completed-output boundary.

    Offsets are zero-based characters or token IDs. ``completed`` excludes the
    malformed frame; slice the original input at ``last_valid_boundary`` for
    recovery. The parser does not retain a second copy of the raw source.
    """

    def __init__(
        self,
        *,
        reason: ParseFailure,
        offset: int,
        unit: Literal["character", "token"],
        index: int,
        last_valid_boundary: int,
        completed: Conversation,
        detail: str = "",
    ) -> None:
        super().__init__(reason, offset, unit, index, detail)
        self.reason = reason
        self.offset = offset
        self.unit = unit
        self.index = index
        self.last_valid_boundary = last_valid_boundary
        self.completed = completed
        self.detail = detail

    def __str__(self) -> str:
        head = (
            f"{self.reason.value} at {self.unit} {self.offset} (message {self.index})"
        )
        return f"{head}: {self.detail}" if self.detail else head


class ParserStateError(RuntimeError):
    """A push follows termination, or a failed parser is reused."""
