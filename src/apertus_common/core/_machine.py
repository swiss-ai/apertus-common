"""The strict output-only state machine over source-neutral lexemes.

The machine accepts exactly the canonical output framing emitted by
``render_str``: output messages separated by one blank line, an optional
trailing wait attached with one newline, or a bare wait. It mutates plain
internal state, never runs Pydantic validation, and reports every rejection
through :class:`MachineFailure` with the absolute source position of the
rejected unit, then stays terminally failed.

Reason assignments not fixed by the design doc: a close token whose frame
shape is wrong (before ``<|hdr|>`` or without a structural space) is
``MISMATCHED_CLOSE``; a non-space character where the structural payload
space belongs, and any text where a message must begin, are
``UNEXPECTED_TEXT``; a wrongly placed wait or out token between messages is
``INVALID_SEPARATOR``; a ``type`` key that is present but not first is
``MISSING_TYPE``; every control token illegal in the current state is
``UNEXPECTED_CONTROL``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from apertus_common.core._lexeme import ControlLexeme, Lexeme, TextLexeme
from apertus_common.core._wire import ControlToken
from apertus_common.errors import HeaderError, ParseFailure
from apertus_common.model._base import HEADER_KEY_RE, check_header_value

_EMPTY_HEADERS: Mapping[str, str] = MappingProxyType({})


class MachineState(StrEnum):
    """Machine states; failure reports carry the value as a plain string."""

    EXPECT_OUTPUT = "expect_output"
    HEADER = "header"
    PAYLOAD = "payload"
    SEPARATOR = "separator"
    WAITING = "waiting"
    FINISHED = "finished"
    FAILED = "failed"


class MachineFailure(Exception):
    """Internal failure record; the facade wraps it into the public error."""

    def __init__(
        self,
        reason: ParseFailure,
        offset: int,
        state: MachineState,
        ordinal: int,
        detail: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(reason, offset, state, ordinal, detail)
        self.reason = reason
        self.offset = offset
        self.state = state
        self.ordinal = ordinal
        self.detail = detail
        self.cause = cause


@dataclass(frozen=True, slots=True)
class ClosedFrame:
    """One syntactically complete output message awaiting construction."""

    ordinal: int
    type: str
    headers: Mapping[str, str]
    payload: str
    start: int
    payload_start: int
    end: int


class OutputMachine:
    """Consumes lexemes, tracks one open frame, and queues close events.

    The payload keeps one candidate character held back until the next
    lexeme: at a close token that character must be the structural trailing
    space and is discarded, otherwise it is emitted as payload. This
    preserves boundary whitespace without normalization and makes chunk
    splitting irrelevant.
    """

    def __init__(self) -> None:
        self._state = MachineState.EXPECT_OUTPUT
        self._at_start = True
        self._count = 0
        self._end = 0
        self._frame_start = 0
        self._header_parts: list[str] = []
        self._header_start = 0
        self._type: str | None = None
        self._headers: dict[str, str] = {}
        self._payload_parts: list[str] = []
        self._held = ""
        self._payload_opened = False
        self._payload_start = 0
        self._separator = ""
        self._waiting = False
        self._emitted: list[tuple[int, str, str]] = []
        self._closed: ClosedFrame | None = None

    @property
    def state(self) -> MachineState:
        """The current machine state."""
        return self._state

    @property
    def waiting(self) -> bool:
        """True once a wait token has been accepted."""
        return self._waiting

    @property
    def current_type(self) -> str | None:
        """The open message's type, once its header has been parsed."""
        return self._type

    @property
    def current_headers(self) -> Mapping[str, str]:
        """The open message's parsed headers excluding ``type``."""
        if self._type is None:
            return _EMPTY_HEADERS
        return MappingProxyType(self._headers)

    @property
    def current_payload(self) -> str | None:
        """The open message's emitted payload, excluding the held character."""
        if self._state is not MachineState.PAYLOAD:
            return None
        return "".join(self._payload_parts)

    def take_emitted(self) -> list[tuple[int, str, str]]:
        """Drain pending ``(ordinal, type, text)`` payload emissions."""
        emitted = self._emitted
        self._emitted = []
        return emitted

    def take_closed(self) -> ClosedFrame | None:
        """Drain the frame closed by the last feed, if any."""
        closed = self._closed
        self._closed = None
        return closed

    def fail(
        self,
        reason: ParseFailure,
        offset: int,
        detail: str,
        cause: BaseException | None = None,
        ordinal: int | None = None,
    ) -> MachineFailure:
        """Enter the failed state on behalf of the facade and return the record."""
        return self._fail(reason, offset, detail, cause=cause, ordinal=ordinal)

    def feed(self, lexeme: Lexeme) -> None:
        """Consume one lexeme; raises :class:`MachineFailure` on rejection."""
        if isinstance(lexeme, ControlLexeme):
            self._feed_control(lexeme)
        else:
            self._feed_text(lexeme)
        self._end = lexeme.span.end

    def finish(self) -> None:
        """End of input; raises unless the machine sits at a clean boundary."""
        state = self._state
        if state is MachineState.WAITING:
            self._state = MachineState.FINISHED
            return
        if state is MachineState.EXPECT_OUTPUT:
            if self._at_start:
                self._state = MachineState.FINISHED
                return
            raise self._fail(
                ParseFailure.INCOMPLETE_SEPARATOR,
                self._end,
                "input ends after a message separator with no next message",
            )
        if state is MachineState.SEPARATOR:
            if not self._separator:
                self._state = MachineState.FINISHED
                return
            raise self._fail(
                ParseFailure.INCOMPLETE_SEPARATOR,
                self._end,
                "input ends inside a message separator",
            )
        if state is MachineState.HEADER:
            raise self._fail(
                ParseFailure.INCOMPLETE_HEADER,
                self._end,
                "input ends inside a message header",
            )
        raise self._fail(
            ParseFailure.INCOMPLETE_PAYLOAD,
            self._end,
            "input ends inside a message payload",
        )

    def _fail(
        self,
        reason: ParseFailure,
        offset: int,
        detail: str,
        cause: BaseException | None = None,
        ordinal: int | None = None,
    ) -> MachineFailure:
        if ordinal is None:
            ordinal = self._count - 1 if self._frame_open() else self._count
        failure = MachineFailure(reason, offset, self._state, ordinal, detail, cause)
        self._state = MachineState.FAILED
        return failure

    def _frame_open(self) -> bool:
        return self._state in (MachineState.HEADER, MachineState.PAYLOAD)

    def _feed_control(self, lexeme: ControlLexeme) -> None:
        token = lexeme.token
        start = lexeme.span.start
        state = self._state
        if state is MachineState.EXPECT_OUTPUT:
            if token is ControlToken.OUT:
                self._open_frame(lexeme.span.start, lexeme.span.end)
                return
            if token is ControlToken.WAIT:
                if self._at_start:
                    self._waiting = True
                    self._state = MachineState.WAITING
                    return
                raise self._fail(
                    ParseFailure.INVALID_SEPARATOR,
                    start,
                    "a wait attaches with one newline, not a blank line",
                )
            raise self._fail(
                ParseFailure.UNEXPECTED_CONTROL,
                start,
                f"{token.value} where an output message must begin",
            )
        if state is MachineState.HEADER:
            if token is ControlToken.HDR:
                self._parse_header(lexeme.span.start)
                self._state = MachineState.PAYLOAD
                self._payload_parts = []
                self._held = ""
                self._payload_opened = False
                return
            if token is ControlToken.OUT_END:
                raise self._fail(
                    ParseFailure.MISMATCHED_CLOSE,
                    start,
                    f"{ControlToken.OUT_END.value} before {ControlToken.HDR.value}",
                )
            raise self._fail(
                ParseFailure.UNEXPECTED_CONTROL,
                start,
                f"{token.value} inside a message header",
            )
        if state is MachineState.PAYLOAD:
            if token is ControlToken.OUT_END:
                self._close_frame(start, lexeme.span.end)
                return
            raise self._fail(
                ParseFailure.UNEXPECTED_CONTROL,
                start,
                f"{token.value} inside a message payload",
            )
        if state is MachineState.SEPARATOR:
            if token is ControlToken.WAIT and self._separator == "\n":
                self._waiting = True
                self._state = MachineState.WAITING
                return
            if token in (ControlToken.WAIT, ControlToken.OUT):
                raise self._fail(
                    ParseFailure.INVALID_SEPARATOR,
                    start,
                    f"{token.value} after {self._separator!r}; messages are"
                    ' separated by "\\n\\n" and a wait attaches with "\\n"',
                )
            raise self._fail(
                ParseFailure.UNEXPECTED_CONTROL,
                start,
                f"{token.value} between output messages",
            )
        raise self._fail(
            ParseFailure.TRAILING_AFTER_WAIT,
            start,
            f"{token.value} after the wait token",
        )

    def _feed_text(self, lexeme: TextLexeme) -> None:
        text = lexeme.text
        start = lexeme.span.start
        state = self._state
        if state is MachineState.HEADER:
            self._header_parts.append(text)
            return
        if state is MachineState.PAYLOAD:
            if not self._payload_opened:
                if text[0] != " ":
                    raise self._fail(
                        ParseFailure.UNEXPECTED_TEXT,
                        start,
                        f"expected the structural space after"
                        f" {ControlToken.HDR.value}, found {text[0]!r}",
                    )
                self._payload_opened = True
                self._payload_start = start + 1
                text = text[1:]
            if text:
                combined = self._held + text
                self._held = combined[-1]
                emitted = combined[:-1]
                if emitted:
                    self._emit(emitted)
            return
        if state is MachineState.SEPARATOR:
            self._feed_separator(text, start)
            return
        if state is MachineState.EXPECT_OUTPUT:
            raise self._fail(
                ParseFailure.UNEXPECTED_TEXT,
                start,
                f"text where an output message must begin: {text[:20]!r}",
            )
        raise self._fail(
            ParseFailure.TRAILING_AFTER_WAIT,
            start,
            f"text after the wait token: {text[:20]!r}",
        )

    def _feed_separator(self, text: str, start: int) -> None:
        index = 0
        while self._separator != "\n\n" and index < len(text):
            if text[index] != "\n":
                raise self._fail(
                    ParseFailure.INVALID_SEPARATOR,
                    start + index,
                    f"expected a newline in the message separator,"
                    f" found {text[index]!r}",
                )
            self._separator += "\n"
            index += 1
        if self._separator == "\n\n":
            self._state = MachineState.EXPECT_OUTPUT
            self._at_start = False
            if index < len(text):
                raise self._fail(
                    ParseFailure.UNEXPECTED_TEXT,
                    start + index,
                    f"text where an output message must begin: {text[index:][:20]!r}",
                )

    def _open_frame(self, start: int, header_start: int) -> None:
        self._state = MachineState.HEADER
        self._frame_start = start
        self._header_start = header_start
        self._header_parts = []
        self._count += 1

    def _emit(self, text: str) -> None:
        assert self._type is not None
        self._payload_parts.append(text)
        self._emitted.append((self._count - 1, self._type, text))

    def _close_frame(self, close_start: int, close_end: int) -> None:
        if not self._payload_opened:
            raise self._fail(
                ParseFailure.MISMATCHED_CLOSE,
                close_start,
                f"missing structural space after {ControlToken.HDR.value}",
            )
        if self._held not in ("", " "):
            raise self._fail(
                ParseFailure.MISMATCHED_CLOSE,
                close_start,
                f"expected the structural space before"
                f" {ControlToken.OUT_END.value}, found {self._held!r}",
            )
        if self._held == " " and not self._payload_parts:
            # two-space body: neither the one-space empty payload nor
            # SP PAYLOAD SP; the renderer never emits it
            raise self._fail(
                ParseFailure.MISMATCHED_CLOSE,
                close_start,
                f"two spaces between {ControlToken.HDR.value} and"
                f" {ControlToken.OUT_END.value}; an empty payload is one space",
            )
        assert self._type is not None
        self._closed = ClosedFrame(
            ordinal=self._count - 1,
            type=self._type,
            headers=self._headers,
            payload="".join(self._payload_parts),
            start=self._frame_start,
            payload_start=self._payload_start,
            end=close_end,
        )
        self._type = None
        self._headers = {}
        self._payload_parts = []
        self._held = ""
        self._payload_opened = False
        self._state = MachineState.SEPARATOR
        self._separator = ""

    def _parse_header(self, hdr_offset: int) -> None:
        raw = "".join(self._header_parts)
        start = self._header_start
        if not raw.startswith(" "):
            raise self._fail(
                ParseFailure.INVALID_HEADER,
                start,
                "the header must begin with one space",
            )
        if not raw.endswith(" ") or raw == " ":
            if raw == " ":
                raise self._fail(
                    ParseFailure.MISSING_TYPE, start + 1, "the header has no fields"
                )
            raise self._fail(
                ParseFailure.INVALID_HEADER,
                hdr_offset,
                f"the header must end with one space before {ControlToken.HDR.value}",
            )
        body = raw[1:-1]
        seen: dict[str, int] = {}
        ordered: list[str] = []
        headers: dict[str, str] = {}
        type_value: str | None = None
        offset = start + 1
        for field in body.split(" "):
            if not field:
                raise self._fail(
                    ParseFailure.INVALID_HEADER,
                    offset,
                    "header fields are separated by exactly one space",
                )
            equals = field.find("=")
            if equals <= 0:
                raise self._fail(
                    ParseFailure.INVALID_HEADER,
                    offset,
                    f"header fields are key=value pairs, found {field!r}",
                )
            key = field[:equals]
            value = field[equals + 1 :]
            if HEADER_KEY_RE.fullmatch(key) is None:
                raise self._fail(
                    ParseFailure.INVALID_HEADER,
                    offset,
                    f"invalid header key {key!r}",
                )
            if key == "direction":
                raise self._fail(
                    ParseFailure.INVALID_HEADER,
                    offset,
                    f"direction is forbidden; {ControlToken.OUT.value} supplies it",
                )
            try:
                check_header_value(value)
            except HeaderError as error:
                raise self._fail(
                    ParseFailure.INVALID_HEADER,
                    offset + equals + 1,
                    f"invalid header value for {key!r}",
                    cause=error,
                ) from error
            if key in seen:
                raise self._fail(
                    ParseFailure.DUPLICATE_HEADER,
                    offset,
                    f"repeated header key {key!r}",
                )
            if not ordered and key != "type":
                raise self._fail(
                    ParseFailure.MISSING_TYPE,
                    offset,
                    "the first header field must be type",
                )
            seen[key] = offset
            ordered.append(key)
            if key == "type":
                type_value = value
            else:
                headers[key] = value
            offset += len(field) + 1
        assert type_value is not None
        if type_value == "tool_call" and (len(ordered) < 2 or ordered[1] != "id"):
            raise self._fail(
                ParseFailure.INVALID_HEADER,
                start + 1,
                "tool_call requires exactly one id immediately after type",
            )
        self._type = type_value
        self._headers = headers
        self._header_parts = []
