"""OutputMachine: strict transitions, header grammar, payload holdback.

Imports private core modules deliberately: the machine is internal and its
exact offsets, states, and frame records are contract details the public
facade builds on.
"""

import pytest

from apertus_common.core._machine import (
    ClosedFrame,
    MachineFailure,
    MachineState,
    OutputMachine,
)
from apertus_common.core._scan import TextLexer
from apertus_common.errors import HeaderError, ParseFailure


def drive(
    text: str,
) -> tuple[OutputMachine, list[tuple[int, str, str]], list[ClosedFrame]]:
    """Feed *text* through a lexer into a fresh machine, collecting drains."""
    machine = OutputMachine()
    lexer = TextLexer()
    emitted: list[tuple[int, str, str]] = []
    closed: list[ClosedFrame] = []
    for lexeme in lexer.push(text):
        machine.feed(lexeme)
        emitted.extend(machine.take_emitted())
        frame = machine.take_closed()
        if frame is not None:
            closed.append(frame)
    return machine, emitted, closed


def fail_of(text: str, *, finish: bool = False) -> MachineFailure:
    with pytest.raises(MachineFailure) as info:
        machine, _, _ = drive(text)
        if finish:
            machine.finish()
    return info.value


def test_closed_frame_records_positions_and_content() -> None:
    text = "<|out|> type=assistant k=v <|hdr|> hi <|/out|>"
    machine, emitted, closed = drive(text)
    frame = closed[0]
    assert frame.ordinal == 0
    assert frame.type == "assistant"
    assert dict(frame.headers) == {"k": "v"}
    assert frame.payload == "hi"
    assert frame.start == 0
    assert frame.payload_start == text.index(" hi ") + 1
    assert frame.end == len(text)
    assert emitted == [(0, "assistant", "hi")]
    assert machine.state is MachineState.SEPARATOR


def test_payload_holdback_preserves_boundary_whitespace() -> None:
    cases = {
        "<|out|> type=assistant <|hdr|> <|/out|>": "",
        "<|out|> type=assistant <|hdr|>   <|/out|>": " ",
        "<|out|> type=assistant <|hdr|>  x  <|/out|>": " x ",
        "<|out|> type=assistant <|hdr|> a\n\nb <|/out|>": "a\n\nb",
        "<|out|> type=assistant <|hdr|> tab\t <|/out|>": "tab\t",
    }
    for text, expected in cases.items():
        _, _, closed = drive(text)
        assert closed[0].payload == expected, text


def test_emitted_payload_excludes_held_character_until_resolved() -> None:
    machine = OutputMachine()
    lexer = TextLexer()
    for lexeme in lexer.push("<|out|> type=think <|hdr|> ab"):
        machine.feed(lexeme)
    assert machine.take_emitted() == [(0, "think", "a")]
    assert machine.current_payload == "a"
    for lexeme in lexer.push("c <|/out|>"):
        machine.feed(lexeme)
    assert machine.take_emitted() == [(0, "think", "bc")]
    assert machine.take_closed() is not None


def test_current_views_by_phase() -> None:
    machine = OutputMachine()
    lexer = TextLexer()

    def feed(text: str) -> None:
        for lexeme in lexer.push(text):
            machine.feed(lexeme)

    feed("<|out|> type=memo k=v ")
    assert machine.current_type is None
    assert dict(machine.current_headers) == {}
    assert machine.current_payload is None
    feed("<|hdr|> pay")
    assert machine.current_type == "memo"
    assert dict(machine.current_headers) == {"k": "v"}
    assert machine.current_payload == "pa"
    feed(" <|/out|>")
    machine.take_closed()
    assert machine.current_type is None
    assert machine.current_payload is None


def test_header_missing_type_offsets() -> None:
    failure = fail_of("<|out|> id=a:1 <|hdr|> x <|/out|>")
    assert failure.reason is ParseFailure.MISSING_TYPE
    assert failure.offset == 8
    empty = fail_of("<|out|> <|hdr|> x <|/out|>")
    assert empty.reason is ParseFailure.MISSING_TYPE


def test_header_duplicate_key_offset_points_at_repeat() -> None:
    text = "<|out|> type=assistant k=v k=w <|hdr|> x <|/out|>"
    failure = fail_of(text)
    assert failure.reason is ParseFailure.DUPLICATE_HEADER
    assert failure.offset == text.index("k=w")
    assert failure.state is MachineState.HEADER


def test_header_value_failure_chains_header_error() -> None:
    failure = fail_of("<|out|> type= <|hdr|> x <|/out|>")
    assert failure.reason is ParseFailure.INVALID_HEADER
    assert isinstance(failure.cause, HeaderError)


def test_header_rejects_direction_and_bad_keys() -> None:
    direction = fail_of("<|out|> type=assistant direction=out <|hdr|> x <|/out|>")
    assert direction.reason is ParseFailure.INVALID_HEADER
    bad_key = fail_of("<|out|> type=assistant so-so=v <|hdr|> x <|/out|>")
    assert bad_key.reason is ParseFailure.INVALID_HEADER
    bare = fail_of("<|out|> type=assistant flag <|hdr|> x <|/out|>")
    assert bare.reason is ParseFailure.INVALID_HEADER


def test_header_spacing_is_strict() -> None:
    double = fail_of("<|out|> type=assistant  <|hdr|> x <|/out|>")
    assert double.reason is ParseFailure.INVALID_HEADER
    missing_lead = fail_of("<|out|>type=assistant <|hdr|> x <|/out|>")
    assert missing_lead.reason is ParseFailure.INVALID_HEADER
    missing_tail = fail_of("<|out|> type=assistant<|hdr|> x <|/out|>")
    assert missing_tail.reason is ParseFailure.INVALID_HEADER


def test_tool_call_requires_id_directly_after_type() -> None:
    late = fail_of("<|out|> type=tool_call k=v id=a:1 <|hdr|> {} <|/out|>")
    assert late.reason is ParseFailure.INVALID_HEADER
    missing = fail_of("<|out|> type=tool_call <|hdr|> {} <|/out|>")
    assert missing.reason is ParseFailure.INVALID_HEADER


def test_type_not_first_is_missing_type() -> None:
    failure = fail_of("<|out|> k=v type=assistant <|hdr|> x <|/out|>")
    assert failure.reason is ParseFailure.MISSING_TYPE


def test_close_without_structural_spaces_is_mismatched() -> None:
    no_space = fail_of("<|out|> type=assistant <|hdr|><|/out|>")
    assert no_space.reason is ParseFailure.MISMATCHED_CLOSE
    no_trailing = fail_of("<|out|> type=assistant <|hdr|> hi<|/out|>")
    assert no_trailing.reason is ParseFailure.MISMATCHED_CLOSE
    before_hdr = fail_of("<|out|> type=assistant <|/out|>")
    assert before_hdr.reason is ParseFailure.MISMATCHED_CLOSE
    two_space_body = fail_of("<|out|> type=assistant <|hdr|>  <|/out|>")
    assert two_space_body.reason is ParseFailure.MISMATCHED_CLOSE


def test_payload_scope_rejects_foreign_controls() -> None:
    for glyph in (
        "<|sys|>",
        "<|in|>",
        "<|/in|>",
        "<|pad|>",
        "<|hdr|>",
        "<|out|>",
    ):
        failure = fail_of(f"<|out|> type=assistant <|hdr|> x {glyph} <|/out|>")
        assert failure.reason is ParseFailure.UNEXPECTED_CONTROL, glyph
        assert failure.state is MachineState.PAYLOAD


def test_separator_grammar() -> None:
    prefix = "<|out|> type=assistant <|hdr|> x <|/out|>"
    triple = fail_of(prefix + "\n\n\nmore")
    assert triple.reason is ParseFailure.UNEXPECTED_TEXT
    assert triple.offset == len(prefix) + 2
    wrong = fail_of(prefix + "\nx")
    assert wrong.reason is ParseFailure.INVALID_SEPARATOR
    early_out = fail_of(prefix + "\n<|out|>")
    assert early_out.reason is ParseFailure.INVALID_SEPARATOR
    early_wait = fail_of(prefix + "<|wait|>")
    assert early_wait.reason is ParseFailure.INVALID_SEPARATOR
    blank_wait = fail_of(prefix + "\n\n<|wait|>")
    assert blank_wait.reason is ParseFailure.INVALID_SEPARATOR


def test_wait_lifecycle_and_trailing() -> None:
    machine, _, _ = drive("<|wait|>")
    assert machine.waiting
    assert machine.state is MachineState.WAITING
    machine.finish()
    assert machine.state is MachineState.FINISHED
    trailing = fail_of("<|out|> type=assistant <|hdr|> x <|/out|>\n<|wait|>x")
    assert trailing.reason is ParseFailure.TRAILING_AFTER_WAIT
    assert trailing.state is MachineState.WAITING


def test_finish_reports_incomplete_reasons() -> None:
    cases = {
        "<|out|> type=assistant": ParseFailure.INCOMPLETE_HEADER,
        "<|out|> type=assistant <|hdr|> x": ParseFailure.INCOMPLETE_PAYLOAD,
        "<|out|> type=assistant <|hdr|> x <|/out|>\n": (
            ParseFailure.INCOMPLETE_SEPARATOR
        ),
        "<|out|> type=assistant <|hdr|> x <|/out|>\n\n": (
            ParseFailure.INCOMPLETE_SEPARATOR
        ),
    }
    for text, reason in cases.items():
        failure = fail_of(text, finish=True)
        assert failure.reason is reason, text
        assert failure.offset == len(text)


def test_clean_finishes() -> None:
    machine, _, _ = drive("")
    machine.finish()
    assert machine.state is MachineState.FINISHED
    machine, _, closed = drive("<|out|> type=assistant <|hdr|> x <|/out|>")
    machine.finish()
    assert machine.state is MachineState.FINISHED
    assert len(closed) == 1


def test_ordinals_increment_per_message() -> None:
    text = (
        "<|out|> type=assistant <|hdr|> a <|/out|>\n\n"
        "<|out|> type=think <|hdr|> b <|/out|>"
    )
    _, emitted, closed = drive(text)
    assert [frame.ordinal for frame in closed] == [0, 1]
    assert {(ordinal, name) for ordinal, name, _ in emitted} == {
        (0, "assistant"),
        (1, "think"),
    }


def test_failure_makes_machine_terminal() -> None:
    machine = OutputMachine()
    lexer = TextLexer()
    with pytest.raises(MachineFailure):
        for lexeme in lexer.push("junk"):
            machine.feed(lexeme)
    assert machine.state is MachineState.FAILED


def test_facade_fail_records_state_and_ordinal() -> None:
    machine, _, _ = drive("<|out|> type=assistant <|hdr|> x <|/out|>")
    failure = machine.fail(ParseFailure.MESSAGE_VALIDATION, 0, "detail", ordinal=0)
    assert failure.state is MachineState.SEPARATOR
    assert failure.ordinal == 0
    assert machine.state is MachineState.FAILED
