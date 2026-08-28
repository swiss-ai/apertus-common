"""The public streaming parser facade over the lexer and output machine.

``StreamParser`` consumes one contiguous model-output segment, exposes
provisional payload deltas and completed typed output messages, and stops
permanently at the first failure. It builds exactly one Pydantic model per
completed message and never chooses a recovery policy; consumers keep the
committed state and decide what to do with the failure.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn, Self, cast

from pydantic import ValidationError

from apertus_common.core._machine import ClosedFrame, MachineFailure, OutputMachine
from apertus_common.core._scan import TextLexer
from apertus_common.errors import (
    FramingError,
    IncompleteMessageError,
    MessageValidationError,
    ParseFailure,
    ParserStateError,
    StreamParseError,
)
from apertus_common.model import (
    Assistant,
    GenericOutput,
    OutputMessage,
    Think,
    ToolCall,
    VerifiableAnswer,
)

_OUTPUT_TYPES: Mapping[str, type[OutputMessage]] = {
    "assistant": Assistant,
    "think": Think,
    "tool_call": ToolCall,
    "verifiable_answer": VerifiableAnswer,
}

_INCOMPLETE_REASONS = frozenset(
    {
        ParseFailure.INCOMPLETE_CONTROL,
        ParseFailure.INCOMPLETE_HEADER,
        ParseFailure.INCOMPLETE_PAYLOAD,
        ParseFailure.INCOMPLETE_SEPARATOR,
    }
)
_VALIDATION_REASONS = frozenset(
    {ParseFailure.INVALID_TOOL_PAYLOAD, ParseFailure.MESSAGE_VALIDATION}
)


@dataclass(frozen=True, slots=True)
class PayloadDelta:
    """One provisional payload fragment of the identified output message."""

    ordinal: int
    type: str
    text: str


class _ToolPayloadError(ValueError):
    """Internal: a tool-call payload violates the strict JSON contract."""


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str) -> object:
    raise ValueError(f"non-finite JSON number {name}")


def _tool_call_data(payload: str) -> dict[str, object]:
    """Parse a tool-call payload as strict JSON with exactly name and args."""
    try:
        data: object = json.loads(
            payload,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except ValueError as error:
        raise _ToolPayloadError(f"tool-call payload rejected: {error}") from error
    if not isinstance(data, dict):
        raise _ToolPayloadError("tool-call payload must be a JSON object")
    # every parsed object passed through _reject_duplicates, so keys are str
    result = cast(dict[str, object], data)
    if set(result) != {"name", "args"}:
        raise _ToolPayloadError("tool-call payload must contain exactly name and args")
    return result


def _construct[T](cls: type[T], **fields: object) -> T:
    # type erasure only: lets object-typed JSON values reach Pydantic, which
    # validates them; ValidationError handling stays with the caller
    return cls(**fields)


class StreamParser:
    """Strict streaming parser for one contiguous model-output segment.

    Feed text with ``push`` in arbitrary chunk sizes, then call ``finish``
    at end of input. Drain provisional fragments with ``take_deltas`` and
    validated messages with ``take_completed``; ``messages`` stays
    cumulative. Any failure raises a :class:`StreamParseError` subclass
    carrying this parser as ``error.parser`` and leaves it permanently
    terminal; recovery always uses a new parser chosen by the consumer.
    """

    def __init__(self) -> None:
        # Concrete TextLexer type: the facade also reads its pending residue,
        # which the source-neutral Lexer protocol deliberately omits.
        self._lexer = TextLexer()
        self._machine = OutputMachine()
        self._messages: list[OutputMessage] = []
        self._completed: list[OutputMessage] = []
        self._deltas: list[PayloadDelta] = []
        self._chunks: list[str] = []
        self._buffer_start = 0
        self._terminal = False

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse *text* with a fresh parser: one push, then finish.

        Returns the finished parser; on failure the raised error carries
        the terminal parser as ``error.parser`` for state inspection.
        """
        parser = cls()
        parser.push(text)
        parser.finish()
        return parser

    @property
    def messages(self) -> tuple[OutputMessage, ...]:
        """Every completed message so far; draining does not remove history."""
        return tuple(self._messages)

    @property
    def current_type(self) -> str | None:
        """The open output message's type, or None when no message is open."""
        return self._machine.current_type

    @property
    def current_headers(self) -> Mapping[str, str]:
        """The open message's headers excluding ``type``; empty when closed."""
        return self._machine.current_headers

    @property
    def current_payload(self) -> str | None:
        """The open message's accumulated payload, or None when closed."""
        return self._machine.current_payload

    @property
    def waiting(self) -> bool:
        """True once the wait token has been accepted."""
        return self._machine.waiting

    def push(self, chunk: str) -> None:
        """Consume one source chunk; chunks may split tokens and messages."""
        self._ensure_active("push")
        self._chunks.append(chunk)
        try:
            for lexeme in self._lexer.push(chunk):
                self._machine.feed(lexeme)
                self._drain_machine()
        except MachineFailure as failure:
            self._raise(failure)

    def finish(self) -> None:
        """Declare end of input; fails unless at a clean message boundary."""
        self._ensure_active("finish")
        try:
            for lexeme in self._lexer.finish():
                self._machine.feed(lexeme)
                self._drain_machine()
            if self._lexer.pending:
                raise self._machine.fail(
                    ParseFailure.INCOMPLETE_CONTROL,
                    self._lexer.pending_start,
                    f"input ends inside the control prefix {self._lexer.pending!r}",
                )
            self._machine.finish()
        except MachineFailure as failure:
            self._raise(failure)
        self._terminal = True

    def take_deltas(self) -> list[PayloadDelta]:
        """Drain provisional payload deltas in source order."""
        deltas = self._deltas
        self._deltas = []
        return deltas

    def take_completed(self) -> list[OutputMessage]:
        """Drain messages completed and validated since the last call."""
        completed = self._completed
        self._completed = []
        return completed

    def _ensure_active(self, operation: str) -> None:
        if self._terminal:
            raise ParserStateError(
                f"{operation}() on a terminal parser; recovery uses a new StreamParser"
            )

    def _drain_machine(self) -> None:
        for ordinal, type_name, text in self._machine.take_emitted():
            self._deltas.append(PayloadDelta(ordinal, type_name, text))
        frame = self._machine.take_closed()
        if frame is not None:
            message = self._build_message(frame)
            self._messages.append(message)
            self._completed.append(message)
            self._commit(frame.end)

    def _build_message(self, frame: ClosedFrame) -> OutputMessage:
        try:
            message_class = _OUTPUT_TYPES.get(frame.type)
            if message_class is None:
                return _construct(
                    GenericOutput,
                    type=frame.type,
                    payload=frame.payload,
                    extras=frame.headers,
                )
            if message_class is ToolCall:
                data = _tool_call_data(frame.payload)
                headers = dict(frame.headers)
                call_id = headers.pop("id")
                return _construct(
                    ToolCall,
                    id=call_id,
                    name=data["name"],
                    args=data["args"],
                    extras=headers,
                )
            return _construct(
                message_class, payload=frame.payload, extras=frame.headers
            )
        except _ToolPayloadError as error:
            raise self._machine.fail(
                ParseFailure.INVALID_TOOL_PAYLOAD,
                frame.payload_start,
                str(error),
                cause=error.__cause__,
                ordinal=frame.ordinal,
            ) from error
        except ValidationError as error:
            raise self._machine.fail(
                ParseFailure.MESSAGE_VALIDATION,
                frame.start,
                f"invalid {frame.type} output message",
                cause=error,
                ordinal=frame.ordinal,
            ) from error

    def _commit(self, end: int) -> None:
        raw = "".join(self._chunks)
        kept = raw[end - self._buffer_start :]
        self._chunks = [kept] if kept else []
        self._buffer_start = end

    def _raise(self, failure: MachineFailure) -> NoReturn:
        self._terminal = True
        raw = "".join(self._chunks)
        unconsumed = raw[failure.offset - self._buffer_start :]
        if failure.reason in _INCOMPLETE_REASONS:
            error_class: type[StreamParseError] = IncompleteMessageError
        elif failure.reason in _VALIDATION_REASONS:
            error_class = MessageValidationError
        else:
            error_class = FramingError
        raise error_class(
            reason=failure.reason,
            parser=self,
            offset=failure.offset,
            message_ordinal=failure.ordinal,
            state=failure.state.value,
            raw_text=raw,
            unconsumed=unconsumed,
            detail=failure.detail,
        ) from failure.cause
