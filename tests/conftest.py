"""Shared conversation fixtures exercising the complete cast."""

from apertus_common import (
    Assistant,
    Attachment,
    Conversation,
    Document,
    Effort,
    Event,
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
    VerifiableAnswer,
    Wait,
)


def full_cast() -> Conversation:
    """One conversation containing every union member and a full system prompt."""
    return Conversation(
        system=SystemPrompt(
            identity="You are Aria, built by Acme.",
            behavior="Be brief.",
            effort=Effort.MEDIUM,
            tools=[
                ToolSpec(
                    name="get_weather",
                    json_schema={"type": "object"},
                    policy="One city per call.",
                )
            ],
            environment="platform: web",
            extra_tags={"deployment_note": "pilot"},
        ),
        items=[
            Harness(payload="Current time: 2026-07-19 14:32."),
            ToolResult(id="get_weather:0", payload='{"tempC":19}'),
            Retrieval(src="internal-docs:7", payload="The staging cluster..."),
            Event(payload="Webhook arrived."),
            Attachment(name="report.pdf", mime="application/pdf", payload="Q2..."),
            User(
                payload=[
                    TextPart(text="What should I wear? "),
                    TextPart(text="I have a light jacket."),
                ]
            ),
            GenericInput(type="calendar_invite", payload="9:00", extras={"tz": "CET"}),
            Think(payload="Check the weather first."),
            ToolCall(
                id="get_weather:1",
                name="get_weather",
                args={"city": "Lisbon", "units": ["C", "F"], "detail": {"wind": True}},
            ),
            Wait(),
            ToolResult(id="get_weather:1", payload='{"tempC":24}'),
            VerifiableAnswer(payload="42"),
            GenericOutput(type="render", payload="<canvas/>"),
            Assistant(payload="Light jacket."),
            Wait(),
        ],
    )


def pretraining() -> Conversation:
    """The pretraining shape: no system prompt, document input."""
    return Conversation(
        items=[
            Document(payload="Once upon a time..."),
            Think(payload="Benign fiction."),
        ]
    )
