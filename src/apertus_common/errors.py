"""Shared exception hierarchy for invalid Apertus format data.

The model package raises :class:`HeaderError` from Pydantic validators when a
header cannot be represented safely. Callers normally receive it inside a
``pydantic.ValidationError``, while direct validation helpers may expose it as
an :class:`ApertusFormatError`. The formats package raises
:class:`AdapterError` for what the OpenAI adapter cannot map; the checker's
``ConformanceError`` shares the base. The streaming parser raises
:class:`StreamParseError` subclasses for malformed model output, each carrying
a stable :class:`ParseFailure` reason and the terminal parser, and
:class:`ParserStateError` for caller misuse of a terminal parser.
Vocabulary-specific subclasses will join this module when the token codec
lands.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from apertus_common.parsing import StreamParser


class ApertusFormatError(Exception):
    """Base class for errors caused by invalid Apertus format data.

    Catch this when a caller wants one library-level exception category rather
    than handling a specific header, framing, or vocabulary failure.
    """


class HeaderError(ApertusFormatError, ValueError):
    """A message header violates the Apertus grammar.

    The :class:`ValueError` base lets Pydantic attach the failure to the
    relevant model field and include it in ``pydantic.ValidationError``.
    """


class AdapterError(ApertusFormatError):
    """The OpenAI adapter could not map a message, item, or system prompt.

    ``from_openai`` raises it with ``where="message"`` and the index into the
    OpenAI message list, or ``where="system"`` (index ``-1``) for an invalid
    effort or tools entry; ``to_openai`` with ``where="item"`` and the index
    into ``Conversation.items``, or ``where="system"`` for the system prompt.
    A chained pydantic ``ValidationError`` or JSON error is the ``__cause__``
    where one exists.
    """

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
    """Stable machine-readable reasons carried by :class:`StreamParseError`.

    Values are part of the public contract: consumers may branch on them to
    select a recovery policy without string-matching error text.
    """

    UNEXPECTED_TEXT = "unexpected_text"
    UNEXPECTED_CONTROL = "unexpected_control"
    INVALID_SEPARATOR = "invalid_separator"
    INVALID_HEADER = "invalid_header"
    DUPLICATE_HEADER = "duplicate_header"
    MISSING_TYPE = "missing_type"
    MISMATCHED_CLOSE = "mismatched_close"
    INVALID_TOOL_PAYLOAD = "invalid_tool_payload"
    MESSAGE_VALIDATION = "message_validation"
    TRAILING_AFTER_WAIT = "trailing_after_wait"
    INCOMPLETE_CONTROL = "incomplete_control"
    INCOMPLETE_HEADER = "incomplete_header"
    INCOMPLETE_PAYLOAD = "incomplete_payload"
    INCOMPLETE_SEPARATOR = "incomplete_separator"


class StreamParseError(ApertusFormatError):
    """The streaming parser rejected model output and became terminal.

    Attributes: ``reason`` (stable :class:`ParseFailure`), ``parser`` (the
    exact terminal ``StreamParser`` with its retained messages, queues, and
    frame views), ``offset`` (absolute source offset of the rejected unit),
    ``message_ordinal`` (zero-based output-message position), ``state``
    (parser state name at failure), ``raw_text`` (complete uncommitted frame
    or fragment), ``unconsumed`` (suffix beginning at the rejected unit), and
    ``detail`` (human-readable specifics). A chained JSON or Pydantic cause
    appears as ``__cause__`` where one exists.
    """

    def __init__(
        self,
        *,
        reason: ParseFailure,
        parser: StreamParser,
        offset: int,
        message_ordinal: int,
        state: str,
        raw_text: str,
        unconsumed: str,
        detail: str = "",
    ) -> None:
        super().__init__(reason, offset, message_ordinal, state, detail)
        self.reason = reason
        self.parser = parser
        self.offset = offset
        self.message_ordinal = message_ordinal
        self.state = state
        self.raw_text = raw_text
        self.unconsumed = unconsumed
        self.detail = detail

    def __str__(self) -> str:
        head = (
            f"{self.reason.value} at offset {self.offset}"
            f" (message {self.message_ordinal}, state {self.state})"
        )
        return f"{head}: {self.detail}" if self.detail else head


class FramingError(StreamParseError):
    """Model output violates the strict canonical output grammar.

    Covers unexpected text or control tokens, malformed headers, malformed
    separators, mismatched close tokens, and content after a wait.
    """


class IncompleteMessageError(FramingError):
    """The source ended inside a control token, header, payload, or separator.

    Raised only by ``StreamParser.finish()``; a clean message boundary or an
    accepted wait finishes without error.
    """


class MessageValidationError(StreamParseError):
    """A syntactically framed output message failed typed construction.

    Covers invalid tool-call payload JSON (reason ``INVALID_TOOL_PAYLOAD``)
    and Pydantic model failures (reason ``MESSAGE_VALIDATION``); the parsing
    or validation cause is chained as ``__cause__``.
    """


class ParserStateError(RuntimeError):
    """Caller misuse of a terminal parser, not malformed model output.

    Raised by ``push()`` after a successful ``finish()``, by a second
    ``finish()``, and by either call on a failed parser.
    """
