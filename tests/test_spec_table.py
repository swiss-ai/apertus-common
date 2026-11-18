"""The typed cast matches the pinned Apertus 2 profile inventory."""

from apertus_common import Call, Claim, Document, Host, Reply, Result, Think, User
from apertus_common.core._wire import ControlToken


def test_known_cast_and_control_inventory():
    inputs = [
        Host(payload=""),
        User(payload=""),
        Document(payload=""),
        Result(name="t", counter=0, payload=""),
    ]
    outputs = [
        Think(payload=""),
        Reply(payload=""),
        Claim(payload=""),
        Call(name="t", counter=0, payload=""),
    ]
    assert {message.type for message in inputs} == {
        "host",
        "user",
        "document",
        "result",
    }
    assert {message.direction for message in inputs} == {"in"}
    assert {message.type for message in outputs} == {"think", "reply", "claim", "call"}
    assert {message.direction for message in outputs} == {"out"}
    assert {token.value for token in ControlToken} == {
        "<|in|>",
        "<|/in|>",
        "<|out|>",
        "<|/out|>",
        "<|hdr|>",
        "<|wait|>",
        "<|pad|>",
    }
