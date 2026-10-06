"""Native message construction separates structural and conformance rules."""

import pytest
from pydantic import ValidationError

from apertus_common import (
    Call,
    Claim,
    Document,
    Host,
    InputMessage,
    Message,
    OutputMessage,
    Reply,
    Result,
    Think,
    UnknownOutput,
    User,
)


@pytest.mark.parametrize("cls", [Host, User, Think, Reply, Claim])
def test_plain_messages_have_only_text_payload(cls):
    message = cls(payload=" exact \n")
    assert message.payload == " exact \n"
    assert Message.from_json(message.to_json()) == message
    with pytest.raises(ValidationError):
        cls(payload=1)
    with pytest.raises(ValidationError):
        cls(payload="x", extras={})


@pytest.mark.parametrize("cls", [Call, Result])
def test_integer_counter_is_structural_but_sign_and_arguments_are_not(cls):
    message = cls(name="weather", counter=-1, payload="not JSON")
    assert message.counter == -1
    for counter in (True, "1", 1.0):
        with pytest.raises(ValidationError):
            cls(name="weather", counter=counter, payload="")


@pytest.mark.parametrize("cls", [Message, InputMessage, OutputMessage])
def test_bases_are_not_constructible(cls):
    with pytest.raises(ValidationError):
        cls(direction="out", type="reply", payload="")


def test_document_optional_provenance_and_unknown_output():
    assert Document(payload="text").from_ is None
    unknown = UnknownOutput(
        type="future", headers={"nested": {"flag": True}}, payload="raw"
    )
    assert Message.from_json(unknown.to_json()) == unknown
    with pytest.raises(ValidationError):
        UnknownOutput(type="call", payload="")


def test_call_authoring_and_builders():
    call = Call.build(name="weather", counter=0, arguments={"city": "Zürich"})
    assert call.payload == '{"city":"Zürich"}'
    assert (
        call
        == Call.builder()
        .name("weather")
        .counter(0)
        .arguments({"city": "Zürich"})
        .build()
    )
    assert User.builder().payload("hi").build() == User(payload="hi")
    with pytest.raises(ValueError, match="exactly one"):
        Call.build(name="weather", counter=0, payload="{}", arguments={})
    with pytest.raises(ValueError, match="exactly one"):
        Call.builder().name("weather").counter(0).payload("{}").arguments({}).build()
