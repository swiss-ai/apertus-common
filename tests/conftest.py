"""Shared native Apertus 2 examples."""

from apertus_common import (
    Call,
    Claim,
    Conversation,
    Document,
    Host,
    Reply,
    Result,
    SystemPrompt,
    Think,
    Thinking,
    ToolSpec,
    UnknownOutput,
    User,
    Wait,
)


def full_cast() -> Conversation:
    return Conversation(
        system=SystemPrompt(
            thinking=Thinking.MEDIUM,
            identity="You are Aria.",
            behavior="Be brief.",
            environment="web",
            tools=[
                ToolSpec(
                    name="weather",
                    description="Weather",
                    json_schema={"type": "object"},
                )
            ],
        ),
        items=[
            Host(payload="Current time: 2026-10-01."),
            Document(payload="Reference", from_="user", id="a1"),
            User(payload="What is the weather?"),
            Think(payload="Check."),
            Call(name="weather", counter=0, payload='{"city":"Zurich"}'),
            Wait(),
            Result(name="weather", counter=0, payload='{"tempC":20}'),
            UnknownOutput(output_type="memo", payload="extra"),
            Claim(payload="20"),
            Reply(payload="20 degrees."),
            Wait(),
        ],
    )
