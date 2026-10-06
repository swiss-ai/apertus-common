"""One source-neutral framing machine shared by text and token frontends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import TypeAdapter

from apertus_common.core._lexeme import ControlLexeme, Lexeme, TextLexeme
from apertus_common.core._wire import ControlToken
from apertus_common.errors import ParseFailure
from apertus_common.model import JsonObject
from apertus_common.model._base import check_type, strict_json_loads
from apertus_common.model._output import validate_header_json

_HEADER_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


class MachineState(StrEnum):
    EXPECT_OUTPUT = "expect_output"
    HEADER = "header"
    PAYLOAD = "payload"
    WAITING = "waiting"
    FINISHED = "finished"
    FAILED = "failed"


class MachineFailure(Exception):
    """Private error converted into a public failure by the parser facade."""

    def __init__(
        self,
        reason: ParseFailure,
        offset: int,
        index: int,
        detail: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(reason, offset, index, detail)
        self.reason, self.offset, self.index = reason, offset, index
        self.detail, self.cause = detail, cause


@dataclass(frozen=True, slots=True)
class StartedFrame:
    index: int
    type: str
    headers: JsonObject


@dataclass(frozen=True, slots=True)
class PayloadPiece:
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class ClosedFrame:
    index: int
    type: str
    headers: JsonObject
    payload: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class WaitFrame:
    end: int


type MachineEvent = StartedFrame | PayloadPiece | ClosedFrame | WaitFrame


def _header(raw: str) -> tuple[str, JsonObject]:
    if any(char.isspace() for char in raw):
        raise ValueError("headers contain no whitespace")
    name, brace, rest = raw.partition("{")
    check_type(name)
    headers = _HEADER_ADAPTER.validate_python(
        strict_json_loads(brace + rest) if brace else {}
    )
    validate_header_json(headers)
    if name in {"think", "reply", "claim"} and brace:
        raise ValueError(f"{name} requires a type-only header")
    if name == "call":
        if set(headers) != {"name", "counter"}:
            raise ValueError("call requires exactly name and counter headers")
        if not isinstance(headers["name"], str):
            raise ValueError("call name must be a string")
        if type(headers["counter"]) is not int:
            raise ValueError("call counter must be an integer")
    return name, headers


class OutputMachine:
    """Track one frame and emit ordered observations without payload checks."""

    def __init__(self) -> None:
        self.state = MachineState.EXPECT_OUTPUT
        self.index = 0
        self.consumed = 0
        self.start = 0
        self.header_start = 0
        self.type: str | None = None
        self.headers: JsonObject = _HEADER_ADAPTER.validate_python({})
        self._header_parts: list[str] = []
        self._payload_parts: list[str] = []

    @property
    def payload(self) -> str:
        return "".join(self._payload_parts)

    def feed(self, lexeme: Lexeme) -> MachineEvent | None:
        if self.state is MachineState.WAITING:
            raise self.fail(
                ParseFailure.TRAILING_AFTER_WAIT,
                lexeme.span.start,
                "wait ends the output segment",
            )
        if self.state in {MachineState.FINISHED, MachineState.FAILED}:
            raise RuntimeError("feed on terminal output machine")
        event: MachineEvent | None = None
        if isinstance(lexeme, TextLexeme):
            if self.state is MachineState.HEADER:
                self._header_parts.append(lexeme.text)
            elif self.state is MachineState.PAYLOAD:
                self._payload_parts.append(lexeme.text)
                if lexeme.text:
                    event = PayloadPiece(self.index, lexeme.text)
            elif lexeme.text:
                raise self.fail(
                    ParseFailure.UNEXPECTED_TEXT,
                    lexeme.span.start,
                    "expected output start or wait",
                )
        else:
            event = self._control(lexeme)
        self.consumed = lexeme.span.end
        return event

    def _control(self, lexeme: ControlLexeme) -> MachineEvent | None:
        token, offset = lexeme.token, lexeme.span.start
        if self.state is MachineState.EXPECT_OUTPUT:
            if token is ControlToken.OUT:
                self.state = MachineState.HEADER
                self.start, self.header_start = offset, lexeme.span.end
                return None
            if token is ControlToken.WAIT:
                self.state = MachineState.WAITING
                return WaitFrame(lexeme.span.end)
        elif self.state is MachineState.HEADER and token is ControlToken.HDR:
            try:
                self.type, self.headers = _header("".join(self._header_parts))
            except (ValueError, RecursionError) as error:
                reason = (
                    ParseFailure.DUPLICATE_HEADER
                    if "duplicate JSON key" in str(error)
                    else ParseFailure.INVALID_HEADER
                )
                raise self.fail(reason, self.header_start, str(error), error) from error
            self._header_parts.clear()
            self.state = MachineState.PAYLOAD
            return StartedFrame(self.index, self.type, self.headers)
        elif self.state is MachineState.PAYLOAD and token is ControlToken.OUT_END:
            assert self.type is not None
            frame = ClosedFrame(
                self.index,
                self.type,
                self.headers,
                self.payload,
                self.start,
                lexeme.span.end,
            )
            self._payload_parts.clear()
            self.type = None
            self.state = MachineState.EXPECT_OUTPUT
            self.index += 1
            return frame
        reason = (
            ParseFailure.MISMATCHED_CLOSE
            if token in {ControlToken.OUT_END, ControlToken.IN_END}
            else ParseFailure.UNEXPECTED_CONTROL
        )
        raise self.fail(
            reason, offset, f"unexpected {token.value} in {self.state.value}"
        )

    def finish(self) -> None:
        if self.state is MachineState.HEADER:
            raise self.fail(
                ParseFailure.INCOMPLETE_HEADER, self.consumed, "header is incomplete"
            )
        if self.state is MachineState.PAYLOAD:
            raise self.fail(
                ParseFailure.INCOMPLETE_PAYLOAD,
                self.consumed,
                "output close is missing",
            )
        self.state = MachineState.FINISHED

    def fail(
        self,
        reason: ParseFailure,
        offset: int,
        detail: str,
        cause: BaseException | None = None,
        *,
        index: int | None = None,
    ) -> MachineFailure:
        self.state = MachineState.FAILED
        return MachineFailure(
            reason, offset, self.index if index is None else index, detail, cause
        )
