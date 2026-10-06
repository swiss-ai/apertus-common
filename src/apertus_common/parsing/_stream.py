"""Lightweight output reconstruction over a shared framing machine."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, NoReturn

from pydantic import ValidationError

from apertus_common.core._lexeme import Lexeme
from apertus_common.core._machine import (
    ClosedFrame,
    MachineFailure,
    MachineState,
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

    def finish(self) -> Conversation:
        if self._failed:
            raise ParserStateError("finish on a failed parser; use a new parser")
        if self._result is None:
            try:
                self._machine.finish()
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

    def _construct(self, frame: ClosedFrame) -> ParsedMessage:
        try:
            message_type = _OUTPUT_TYPES.get(frame.type)
            if message_type is None:
                return UnknownOutput(
                    type=frame.type, headers=frame.headers, payload=frame.payload
                )
            return message_type.model_validate(
                {**frame.headers, "payload": frame.payload}
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

    def finish(self) -> Conversation:
        if self._result is not None or self._failed:
            return super().finish()
        self.feed(self._lexer.finish())
        if self._lexer.pending:
            self.fail(
                ParseFailure.INCOMPLETE_CONTROL,
                self._lexer.pending_start,
                "input ends inside a control spelling",
            )
        return super().finish()


def parse_output_text(text: str) -> Conversation:
    """Parse one complete output segment without running conformance checks."""
    parser = TextStreamParser()
    parser.push_text(text)
    return parser.finish()
