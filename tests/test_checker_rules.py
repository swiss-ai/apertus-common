"""One violating fixture per active rule id, plus conforming
counterparts where the boundary is not obvious."""

from apertus_common import (
    Assistant,
    Checker,
    Conversation,
    Harness,
    IdConventionChecker,
    InputOrderChecker,
    Location,
    Severity,
    SystemPrompt,
    SystemPromptChecker,
    Think,
    ToolCall,
    ToolCallChecker,
    ToolResult,
    ToolResultChecker,
    ToolSpec,
    User,
    VerifiableAnswer,
    VerifiableAnswerChecker,
    Violation,
    Wait,
    WaitChecker,
    check,
)

_WEATHER_TOOLS = [
    ToolSpec(
        name="get_weather",
        json_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )
]


def _only(report_rule: str, conversation: Conversation, checker: Checker) -> Violation:
    report = check(conversation, checkers=[checker])
    matching = [v for v in report if v.rule == report_rule]
    assert len(matching) == 1, f"expected one {report_rule}, got {list(report)}"
    return matching[0]


def test_input_order_canonical_order() -> None:
    bad = Conversation(items=[User(payload="u"), Harness(payload="h")])
    violation = _only("input-order/canonical-order", bad, InputOrderChecker())
    assert violation.severity is Severity.ERROR
    assert violation.location == Location(item=1)
    good = Conversation(items=[Harness(payload="h"), User(payload="u")])
    assert check(good, checkers=[InputOrderChecker()]).ok


def test_tool_call_undeclared_name() -> None:
    bad = Conversation(items=[ToolCall(id="bash:0", name="bash", args={})])
    violation = _only("tool-call/undeclared-name", bad, ToolCallChecker())
    assert violation.severity is Severity.ERROR
    assert violation.location == Location(item=0)
    good = Conversation(
        system=SystemPrompt(effort="low", tools=_WEATHER_TOOLS),
        items=[ToolCall(id="get_weather:0", name="get_weather", args={"city": "Bern"})],
    )
    assert check(good, checkers=[ToolCallChecker()]).ok


def test_tool_call_args_invalid() -> None:
    bad = Conversation(
        system=SystemPrompt(effort="low", tools=_WEATHER_TOOLS),
        items=[ToolCall(id="get_weather:0", name="get_weather", args={"city": 3})],
    )
    violation = _only("tool-call/args-invalid", bad, ToolCallChecker())
    assert violation.severity is Severity.ERROR
    assert violation.location == Location(item=0, field="args")
    assert "city" in violation.message


def test_tool_call_broken_schema() -> None:
    bad = Conversation(
        system=SystemPrompt(
            effort="low",
            tools=[ToolSpec(name="bash", json_schema={"type": 42})],
        ),
        items=[ToolCall(id="bash:0", name="bash", args={})],
    )
    violation = _only("tool-call/broken-schema", bad, ToolCallChecker())
    assert violation.severity is Severity.ERROR
    assert violation.location == Location(scope="system", field="tools[0].json_schema")
    # declaration-gated: fires even when the broken tool is never called
    uncalled = Conversation(system=bad.system, items=[User(payload="hi")])
    assert _only("tool-call/broken-schema", uncalled, ToolCallChecker())
    # args of calls to the broken tool are skipped, not reported
    assert {v.rule for v in check(bad, checkers=[ToolCallChecker()])} == {
        "tool-call/broken-schema"
    }

    unknown_dialect = Conversation(
        system=SystemPrompt(
            effort="low",
            tools=[
                ToolSpec(
                    name="bash",
                    json_schema={
                        "$schema": "https://example.invalid/custom-dialect",
                        "type": "object",
                    },
                )
            ],
        )
    )
    assert (
        "unsupported JSON Schema dialect"
        in _only("tool-call/broken-schema", unknown_dialect, ToolCallChecker()).message
    )

    duplicate_with_broken_schema = Conversation(
        system=SystemPrompt(
            effort="low",
            tools=[
                ToolSpec(name="bash", json_schema={"type": "object"}),
                ToolSpec(name="bash", json_schema={"type": 42}),
            ],
        )
    )
    duplicate_violation = _only(
        "tool-call/broken-schema", duplicate_with_broken_schema, ToolCallChecker()
    )
    assert duplicate_violation.location == Location(
        scope="system", field="tools[1].json_schema"
    )

    unresolved_reference = Conversation(
        system=SystemPrompt(
            effort="low",
            tools=[
                ToolSpec(
                    name="bash",
                    json_schema={"$ref": "https://example.invalid/tool-schema"},
                )
            ],
        ),
        items=[ToolCall(id="bash:0", name="bash", args={})],
    )
    reference_violation = _only(
        "tool-call/broken-schema", unresolved_reference, ToolCallChecker()
    )
    assert reference_violation.location == Location(
        scope="system", field="tools[0].json_schema"
    )


def test_tool_call_known_dialect_with_local_reference() -> None:
    conversation = Conversation(
        system=SystemPrompt(
            effort="low",
            tools=[
                ToolSpec(
                    name="weather",
                    json_schema={
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "definitions": {"city": {"type": "string"}},
                        "type": "object",
                        "properties": {"city": {"$ref": "#/definitions/city"}},
                        "required": ["city"],
                    },
                )
            ],
        ),
        items=[ToolCall(id="weather:0", name="weather", args={"city": "Bern"})],
    )
    assert check(conversation, checkers=[ToolCallChecker()]).ok


def test_tool_result_unknown_id() -> None:
    bad = Conversation(items=[ToolResult(id="bash:9", payload="out")])
    violation = _only("tool-result/unknown-id", bad, ToolResultChecker())
    assert violation.severity is Severity.ERROR
    assert violation.location == Location(item=0)


def test_tool_result_duplicate() -> None:
    bad = Conversation(
        items=[
            ToolCall(id="bash:0", name="bash", args={}),
            ToolResult(id="bash:0", payload="out"),
            ToolResult(id="bash:0", payload="again"),
        ]
    )
    violation = _only("tool-result/duplicate", bad, ToolResultChecker())
    assert violation.severity is Severity.ERROR
    assert violation.location == Location(item=2)


def test_tool_result_open_call() -> None:
    bad = Conversation(items=[ToolCall(id="bash:0", name="bash", args={})])
    violation = _only("tool-result/open-call", bad, ToolResultChecker())
    assert violation.severity is Severity.WARNING
    assert violation.location == Location(item=0)
    closed = Conversation(
        items=[
            ToolCall(id="bash:0", name="bash", args={}),
            ToolResult(id="bash:0", payload="out"),
        ]
    )
    assert len(check(closed, checkers=[ToolResultChecker()])) == 0


def test_id_non_canonical() -> None:
    bad = Conversation(items=[ToolCall(id="call_abc123", name="f", args={})])
    violation = _only("tool-call-id/non-canonical", bad, IdConventionChecker())
    assert violation.severity is Severity.WARNING
    # non-canonical ids skip the remaining sub-rules
    assert len(check(bad, checkers=[IdConventionChecker()])) == 1


def test_id_name_mismatch() -> None:
    bad = Conversation(items=[ToolCall(id="bash:0", name="python", args={})])
    violation = _only("tool-call-id/name-mismatch", bad, IdConventionChecker())
    assert violation.severity is Severity.WARNING


def test_id_duplicate_counter() -> None:
    bad = Conversation(
        items=[
            ToolCall(id="bash:0", name="bash", args={}),
            ToolCall(id="python:0", name="python", args={}),
        ]
    )
    violation = _only("tool-call-id/duplicate-counter", bad, IdConventionChecker())
    assert violation.severity is Severity.WARNING
    assert violation.location == Location(item=1)


def test_id_non_sequential() -> None:
    bad = Conversation(
        items=[
            ToolCall(id="bash:0", name="bash", args={}),
            ToolCall(id="bash:5", name="bash", args={}),
        ]
    )
    violation = _only("tool-call-id/non-sequential", bad, IdConventionChecker())
    assert violation.severity is Severity.WARNING
    sequential = Conversation(
        items=[
            ToolCall(id="bash:7", name="bash", args={}),  # first is unconstrained
            ToolCall(id="python:8", name="python", args={}),
        ]
    )
    assert len(check(sequential, checkers=[IdConventionChecker()])) == 0


def test_wait_unterminated_transcript() -> None:
    bad = Conversation(
        items=[User(payload="hi"), Assistant(payload="a"), Wait(), Think(payload="t")]
    )
    violation = _only("wait/unterminated-transcript", bad, WaitChecker())
    assert violation.severity is Severity.WARNING
    assert violation.location == Location(item=3)
    # silent without any wait (pretraining shape)
    no_wait = Conversation(items=[User(payload="hi"), Think(payload="t")])
    assert len(check(no_wait, checkers=[WaitChecker()])) == 0


def test_verifiable_answer_after_assistant() -> None:
    bad = Conversation(
        items=[Assistant(payload="a"), VerifiableAnswer(payload="42"), Wait()]
    )
    violation = _only(
        "verifiable-answer/after-assistant", bad, VerifiableAnswerChecker()
    )
    assert violation.severity is Severity.WARNING
    good = Conversation(
        items=[VerifiableAnswer(payload="42"), Assistant(payload="a"), Wait()]
    )
    assert len(check(good, checkers=[VerifiableAnswerChecker()])) == 0


def test_verifiable_answer_multiple() -> None:
    bad = Conversation(
        items=[
            VerifiableAnswer(payload="41"),
            VerifiableAnswer(payload="42"),
            Wait(),
            VerifiableAnswer(payload="43"),  # next burst is counted separately
            Wait(),
        ]
    )
    violation = _only("verifiable-answer/multiple", bad, VerifiableAnswerChecker())
    assert violation.severity is Severity.WARNING
    assert violation.location == Location(item=1)


def test_system_duplicate_tool_name() -> None:
    bad = Conversation(
        system=SystemPrompt(
            effort="low",
            tools=[
                ToolSpec(name="bash", json_schema={"type": "object"}),
                ToolSpec(name="bash", json_schema={}),
            ],
        )
    )
    violation = _only("system/duplicate-tool-name", bad, SystemPromptChecker())
    assert violation.severity is Severity.ERROR
    assert violation.location == Location(scope="system", field="tools[1]")
