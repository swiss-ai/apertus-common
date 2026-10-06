"""The source-neutral framing machine accepts character and token spans."""

import pytest

from apertus_common.core._lexeme import ControlLexeme, SourceSpan, TextLexeme
from apertus_common.core._machine import (
    ClosedFrame,
    MachineFailure,
    OutputMachine,
    PayloadPiece,
    StartedFrame,
    WaitFrame,
)
from apertus_common.core._wire import ControlToken
from apertus_common.errors import ParseFailure


def test_token_spans_and_literal_control_payload_are_source_neutral():
    machine = OutputMachine()
    observations = []
    for index, value in enumerate(
        [
            ControlToken.OUT,
            "reply",
            ControlToken.HDR,
            "<|wait|>",
            ControlToken.OUT_END,
            ControlToken.WAIT,
        ]
    ):
        span = SourceSpan(index, index + 1, "token")
        lexeme = (
            ControlLexeme(value, span)
            if isinstance(value, ControlToken)
            else TextLexeme(value, span)
        )
        event = machine.feed(lexeme)
        if event is not None:
            observations.append(event)
    assert observations == [
        StartedFrame(0, "reply", {}),
        PayloadPiece(0, "<|wait|>"),
        ClosedFrame(0, "reply", {}, "<|wait|>", 0, 5),
        WaitFrame(6),
    ]
    assert machine.consumed == 6
    machine.finish()


def test_invalid_header_offset_uses_token_header_start():
    machine = OutputMachine()
    machine.feed(ControlLexeme(ControlToken.OUT, SourceSpan(0, 1, "token")))
    machine.feed(
        TextLexeme('call{"counter":true,"name":"x"}', SourceSpan(1, 9, "token"))
    )
    with pytest.raises(MachineFailure) as caught:
        machine.feed(ControlLexeme(ControlToken.HDR, SourceSpan(9, 10, "token")))
    assert caught.value.reason is ParseFailure.INVALID_HEADER
    assert caught.value.offset == 1
    assert machine.consumed == 9
