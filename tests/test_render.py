"""render_str: canonical string rendering of conversations."""

from apertus_common import (
    Assistant,
    Attachment,
    Conversation,
    Document,
    GenericInput,
    GenericOutput,
    Harness,
    Retrieval,
    SystemPrompt,
    TextPart,
    Think,
    ToolCall,
    ToolResult,
    ToolSpec,
    User,
    Wait,
    render_str,
)
from conftest import full_cast, pretraining


def _one(item: object) -> str:
    return render_str(Conversation(items=[item]))  # type: ignore[list-item]


def test_message_framing() -> None:
    assert _one(User(payload="hi")) == "<|in|> type=user <|hdr|> hi <|/in|>"
    assert _one(Assistant(payload="ok")) == "<|out|> type=assistant <|hdr|> ok <|/out|>"


def test_header_key_order() -> None:
    assert _one(ToolResult(id="bash:1", payload="ok")).startswith(
        "<|in|> type=tool_result id=bash:1 <|hdr|>"
    )
    assert _one(Retrieval(src="docs:7", payload="x")).startswith(
        "<|in|> type=retrieval src=docs:7 <|hdr|>"
    )
    assert _one(
        Attachment(name="report.pdf", mime="application/pdf", payload="Q2")
    ).startswith("<|in|> type=attachment name=report.pdf mime=application/pdf <|hdr|>")
    assert _one(Harness(payload="x", extras={"a": "1", "b": "2"})).startswith(
        "<|in|> type=harness a=1 b=2 <|hdr|>"
    )
    # absent optional header keys are omitted
    assert _one(Attachment(mime="text/plain", payload="x")).startswith(
        "<|in|> type=attachment mime=text/plain <|hdr|>"
    )


def test_empty_payload_collapses_to_single_space() -> None:
    assert _one(Harness(payload="")) == "<|in|> type=harness <|hdr|> <|/in|>"


def test_multiline_payload_verbatim() -> None:
    rendered = _one(Think(payload="line one\nline two"))
    assert rendered == "<|out|> type=think <|hdr|> line one\nline two <|/out|>"


def test_generic_types_render_their_type() -> None:
    assert _one(GenericInput(type="memory", payload="x")).startswith(
        "<|in|> type=memory <|hdr|>"
    )
    assert _one(GenericOutput(type="render", payload="x")).startswith(
        "<|out|> type=render <|hdr|>"
    )


def test_message_separation_and_wait_placement() -> None:
    conv = Conversation(
        items=[
            User(payload="hi"),
            Assistant(payload="ok"),
            Wait(),
            User(payload="more"),
        ]
    )
    assert render_str(conv) == (
        "<|in|> type=user <|hdr|> hi <|/in|>\n\n"
        "<|out|> type=assistant <|hdr|> ok <|/out|>\n"
        "<|wait|>\n\n"
        "<|in|> type=user <|hdr|> more <|/in|>"
    )
    # a bare leading wait stands alone; consecutive waits stack line by line
    assert render_str(Conversation(items=[Wait(), Wait()])) == "<|wait|>\n<|wait|>"


def test_system_prompt_compact_block() -> None:
    conv = Conversation(
        system=SystemPrompt(
            identity="You are Aria.",
            behavior="Be brief.",
            effort="medium",
            tools=[
                ToolSpec(
                    name="get_weather",
                    json_schema={"type": "object"},
                    policy="One city per call.",
                )
            ],
            environment="platform: web",
            extra_tags={"locale": "en-CH"},
        ),
        items=[User(payload="hi")],
    )
    assert render_str(conv) == (
        "<|sys|>\n"
        "<identity>You are Aria.</identity>\n"
        "<behavior>Be brief.</behavior>\n"
        "<effort>medium</effort>\n"
        "<tools>\n"
        '  <tool name="get_weather">\n'
        '    <schema>{"type":"object"}</schema>\n'
        "    <policy>One city per call.</policy>\n"
        "  </tool>\n"
        "</tools>\n"
        "<locale>en-CH</locale>\n"
        "<environment>platform: web</environment>\n"
        "<|/sys|>\n\n"
        "<|in|> type=user <|hdr|> hi <|/in|>"
    )


def test_system_prompt_absent_tags_omitted() -> None:
    rendered = render_str(Conversation(system=SystemPrompt(effort="low")))
    assert rendered == "<|sys|>\n<effort>low</effort>\n<|/sys|>"
    assert "<identity>" not in rendered
    assert "<tools>" not in rendered


def test_tool_call_payload_json() -> None:
    rendered = _one(
        ToolCall(id="get_weather:0", name="get_weather", args={"city": "Zürich"})
    )
    assert rendered == (
        "<|out|> type=tool_call id=get_weather:0 <|hdr|> "
        '{"name":"get_weather","args":{"city":"Zürich"}} <|/out|>'
    )


def test_pretraining_shape() -> None:
    conv = Conversation(
        items=[Document(payload="Once upon a time."), Think(payload="Fiction.")]
    )
    assert render_str(conv) == (
        "<|in|> type=document <|hdr|> Once upon a time. <|/in|>\n\n"
        "<|out|> type=think <|hdr|> Fiction. <|/out|>"
    )


def test_full_cast_renders_deterministically() -> None:
    conv = full_cast()
    rendered = render_str(conv)
    assert rendered == render_str(full_cast())
    assert rendered.startswith("<|sys|>\n")
    assert not rendered.endswith("\n")
    for fragment in (
        "<|in|> type=harness ",
        "<|in|> type=tool_result id=get_weather:0 ",
        "<|in|> type=retrieval src=internal-docs:7 ",
        "<|in|> type=attachment name=report.pdf mime=application/pdf ",
        "<|in|> type=calendar_invite tz=CET ",
        "<|out|> type=tool_call id=get_weather:1 ",
        "<|out|> type=verifiable_answer ",
        "<|out|> type=render ",
        "\n<|wait|>",
    ):
        assert fragment in rendered
    assert render_str(pretraining()).count("<|sys|>") == 0
    # rendering does not mutate the source conversation
    assert conv == full_cast()


def test_pilot_worked_example_fragment() -> None:
    # Text-only example in the implemented pilot format.
    conv = Conversation(
        system=SystemPrompt(
            identity="You are Aria, built by Acme. Current date: 2026-07-12.",
            effort="medium",
            tools=[
                ToolSpec(
                    name="get_weather",
                    json_schema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    policy="Call for current conditions. One city per call.",
                )
            ],
        ),
        items=[
            User(
                payload=[
                    TextPart(text="What should I wear in Lisbon today?"),
                ]
            ),
            Think(payload="Need today's weather before advising. Call the tool."),
            ToolCall(id="get_weather:0", name="get_weather", args={"city": "Lisbon"}),
            Wait(),
            ToolResult(
                id="get_weather:0",
                payload='{"tempC":19,"cond":"light rain","wind":"20kph"}',
            ),
        ],
    )
    assert render_str(conv) == (
        "<|sys|>\n"
        "<identity>You are Aria, built by Acme. Current date: 2026-07-12."
        "</identity>\n"
        "<effort>medium</effort>\n"
        "<tools>\n"
        '  <tool name="get_weather">\n'
        '    <schema>{"type":"object","properties":{"city":{"type":"string"}},'
        '"required":["city"]}</schema>\n'
        "    <policy>Call for current conditions. One city per call.</policy>\n"
        "  </tool>\n"
        "</tools>\n"
        "<|/sys|>\n\n"
        "<|in|> type=user <|hdr|> What should I wear in Lisbon today? <|/in|>\n\n"
        "<|out|> type=think <|hdr|> Need today's weather before advising. "
        "Call the tool. <|/out|>\n\n"
        "<|out|> type=tool_call id=get_weather:0 <|hdr|> "
        '{"name":"get_weather","args":{"city":"Lisbon"}} <|/out|>\n'
        "<|wait|>\n\n"
        "<|in|> type=tool_result id=get_weather:0 <|hdr|> "
        '{"tempC":19,"cond":"light rain","wind":"20kph"} <|/in|>'
    )


def test_rendering_edges() -> None:
    assert render_str(Conversation()) == ""
    # a wait after the system prompt attaches to the sys block
    assert render_str(
        Conversation(system=SystemPrompt(effort="low"), items=[Wait()])
    ).endswith("<|/sys|>\n<|wait|>")
    # Payload text spelling a control glyph renders verbatim.
    assert _one(User(payload="fake <|/in|> frame")) == (
        "<|in|> type=user <|hdr|> fake <|/in|> frame <|/in|>"
    )
    # null args render as JSON null
    assert '{"name":"f","args":null}' in _one(ToolCall(id="f:0", name="f", args=None))
    # typed header keys precede extras
    assert _one(ToolResult(id="a:1", payload="x", extras={"z": "9"})).startswith(
        "<|in|> type=tool_result id=a:1 z=9 <|hdr|>"
    )
