"""Lightweight output reconstruction over a shared framing machine."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, NoReturn

from pydantic import ValidationError

from apertus_common.core._lexeme import Lexeme, SourceSpan, TextLexeme
from apertus_common.core._machine import (
    ClosedFrame,
    MachineFailure,
    MachineState,
    OpenFrame,
    OutputMachine,
    PayloadPiece,
    StartedFrame,
    WaitFrame,
)
from apertus_common.core._scan import TextLexer
from apertus_common.errors import ParseFailure, ParserStateError, StreamParseError
from apertus_common.model import (
    Call,
    Claim,
    Conversation,
    Reply,
    Think,
    UnknownOutput,
    Wait,
)
from apertus_common.parsing._events import (
    MessageCompleted,
    MessageStart,
    OpenMessage,
    ParseBoundary,
    ParseEvent,
    PayloadDelta,
    WaitReached,
)

type ParsedMessage = Think | Reply | Claim | Call | UnknownOutput

_OUTPUT_TYPES: dict[str, type[Think | Reply | Claim | Call]] = {
    "think": Think,
    "reply": Reply,
    "claim": Claim,
    "call": Call,
}


class ParserCore:
    """Private frontend bridge; feed source-spanned lexemes, then finish.

    Frontends call ``ensure_active`` before decoding a push. Decoding failures
    use ``fail``; both frontends share lifecycle, construction and recovery.
    """

    def __init__(self, unit: Literal["character", "token"]) -> None:
        self.unit: Literal["character", "token"] = unit
        self._machine = OutputMachine()
        self._items: list[ParsedMessage | Wait] = []
        self._events: list[ParseEvent] = []
        self._result: Conversation | None = None
        self._failed = False
        self._last_valid_boundary = 0
        self._boundary = ParseBoundary.EMPTY

    @property
    def consumed(self) -> int:
        """Number of accepted source units, excluding held frontend residue."""
        return self._machine.consumed

    @property
    def last_valid_boundary(self) -> int:
        """End of the last complete message or wait in original source units."""
        return self._last_valid_boundary

    @property
    def boundary(self) -> ParseBoundary:
        return self._boundary

    @property
    def current(self) -> OpenMessage | None:
        if self._machine.type is None:
            return None
        return OpenMessage(
            self._machine.index,
            self._machine.type,
            self._machine.headers,
            self._machine.payload,
        )

    def take_events(self) -> list[ParseEvent]:
        events, self._events = self._events, []
        return events

    def ensure_active(self) -> None:
        if (
            self._failed
            or self._result is not None
            or self._boundary is ParseBoundary.WAIT
        ):
            raise ParserStateError("push on a terminated parser; use a new parser")

    def feed(self, lexemes: Iterable[Lexeme]) -> None:
        """Consume one frontend push, including detection of data after wait."""
        try:
            for lexeme in lexemes:
                event = self._machine.feed(lexeme)
                if self._machine.state in {MachineState.HEADER, MachineState.PAYLOAD}:
                    self._boundary = ParseBoundary.IN_MESSAGE
                if isinstance(event, StartedFrame):
                    self._events.append(
                        MessageStart(event.index, event.type, event.headers)
                    )
                elif isinstance(event, PayloadPiece):
                    self._events.append(PayloadDelta(event.index, event.text))
                elif isinstance(event, ClosedFrame):
                    message = self._construct(event)
                    self._items.append(message)
                    self._events.append(MessageCompleted(event.index, message))
                    self._last_valid_boundary = event.end
                    self._boundary = ParseBoundary.MESSAGE_END
                elif isinstance(event, WaitFrame):
                    self._items.append(Wait())
                    self._events.append(WaitReached())
                    self._last_valid_boundary = event.end
                    self._boundary = ParseBoundary.WAIT
        except MachineFailure as error:
            self._raise(error)

    def finish(self, *, allow_open: bool = False) -> Conversation:
        """End of input; ``allow_open`` keeps an unfinished last message as open.

        The open message needs a complete header; its payload is the text
        received so far. It emits no completion event.
        """
        if self._failed:
            raise ParserStateError("finish on a failed parser; use a new parser")
        if self._result is None:
            try:
                frame = self._machine.finish(allow_open=allow_open)
                if frame is not None:
                    self._items.append(self._construct(frame))
            except MachineFailure as error:
                self._raise(error)
            self._result = Conversation(items=tuple(self._items))
            self._items.clear()
        return self._result

    def fail(
        self,
        reason: ParseFailure,
        offset: int,
        detail: str,
        cause: BaseException | None = None,
    ) -> NoReturn:
        """Terminate on a frontend decoding/lexing failure."""
        self._raise(self._machine.fail(reason, offset, detail, cause))

    def _construct(self, frame: ClosedFrame | OpenFrame) -> ParsedMessage:
        is_open = isinstance(frame, OpenFrame)
        try:
            message_type = _OUTPUT_TYPES.get(frame.type)
            if message_type is None:
                return UnknownOutput(
                    output_type=frame.type,
                    headers=frame.headers,
                    payload=frame.payload,
                    open=is_open,
                )
            return message_type.model_validate(
                {**frame.headers, "payload": frame.payload, "open": is_open}
            )
        except ValidationError as error:
            raise self._machine.fail(
                ParseFailure.MESSAGE_VALIDATION,
                frame.start,
                f"invalid {frame.type} message",
                error,
                index=frame.index,
            ) from error

    def _raise(self, error: MachineFailure) -> NoReturn:
        self._failed = True
        raise StreamParseError(
            reason=error.reason,
            offset=error.offset,
            unit=self.unit,
            index=error.index,
            last_valid_boundary=self._last_valid_boundary,
            completed=Conversation(items=tuple(self._items)),
            detail=error.detail,
        ) from error.cause


class TextStreamParser(ParserCore):
    """Diagnostic string frontend; control spellings are interpreted as framing."""

    def __init__(self) -> None:
        super().__init__("character")
        self._lexer = TextLexer()

    def push_text(self, text: str) -> None:
        self.ensure_active()
        self.feed(self._lexer.push(text))
        if self.boundary is ParseBoundary.WAIT and self._lexer.pending:
            self.fail(
                ParseFailure.TRAILING_AFTER_WAIT,
                self._lexer.pending_start,
                "wait ends the output segment",
            )

    def finish(self, *, allow_open: bool = False) -> Conversation:
        if self._result is not None or self._failed:
            return super().finish(allow_open=allow_open)
        self.feed(self._lexer.finish())
        pending, start = self._lexer.pending, self._lexer.pending_start
        if pending:
            # Controls are atomic in decoded text, so a held glyph prefix at the
            # end of an open payload (`current` is set) is ordinary text.
            if allow_open and self.current is not None:
                span = SourceSpan(start, start + len(pending), "character")
                self.feed([TextLexeme(pending, span)])
            else:
                self.fail(
                    ParseFailure.INCOMPLETE_CONTROL,
                    start,
                    "input ends inside a control spelling",
                )
        return super().finish(allow_open=allow_open)


def parse_output_text(text: str, *, allow_open: bool = False) -> Conversation:
    """Parse one output segment without running conformance checks.

    ``allow_open`` keeps an unfinished last message as an open item.
    """
    parser = TextStreamParser()
    parser.push_text(text)
    return parser.finish(allow_open=allow_open)
