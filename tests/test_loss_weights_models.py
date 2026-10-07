"""Native training annotations are explicit, strict, and storage-only."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from apertus_common import (
    Call,
    Conversation,
    Message,
    Reply,
    SystemPrompt,
    ToolSpec,
    User,
    Wait,
)


@pytest.mark.parametrize("weight", [True, False, "1", -0.1, float("inf"), float("nan")])
@pytest.mark.parametrize("kind", ["message", "system", "wait"])
def test_loss_weights_reject_non_numeric_negative_and_nonfinite_values(weight, kind):
    with pytest.raises(ValidationError):
        if kind == "message":
            Reply(payload="answer", loss_weight=weight)
        elif kind == "system":
            SystemPrompt.build(loss_weight=weight)
        else:
            Wait(loss_weight=weight)


@pytest.mark.parametrize("weight", [None, 0, 0.25, 2])
def test_loss_weight_storage_roundtrip_and_omission(weight):
    record = Conversation(
        system=SystemPrompt.build(loss_weight=weight),
        items=[User(payload="input", loss_weight=weight), Wait(loss_weight=weight)],
    )
    dumped = json.loads(record.to_json())
    assert dumped["schema_version"] == 2
    for message in [dumped["system"], *dumped["items"]]:
        assert ("loss_weight" in message) == (weight is not None)
        if weight is not None:
            assert message["loss_weight"] == weight
    assert Conversation.from_json(record.to_json()) == record
    assert Message.from_json(record.items[0].to_json()) == record.items[0]
    assert Draft202012Validator(Conversation.model_json_schema()).is_valid(dumped)


def test_v1_native_loading_upgrades_without_rewriting_messages():
    record = Conversation(system=SystemPrompt.build(), items=[Reply(payload="exact\n")])
    old = json.loads(record.to_json())
    old["schema_version"] = 1
    snapshot = Path(__file__).parent / "fixtures" / "native-v1.schema.json"
    assert Draft202012Validator(json.loads(snapshot.read_text())).is_valid(old)
    loaded = Conversation.from_json(json.dumps(old))
    assert loaded == record
    assert json.loads(loaded.to_json())["schema_version"] == 2


@pytest.mark.parametrize("target", ["system", "message", "wait"])
@pytest.mark.parametrize("weight", [None, 0, 1])
def test_v1_rejects_new_annotation_even_when_null(target, weight):
    record = {
        "schema_version": 1,
        "system": {"thinking": "medium"},
        "items": [
            {"direction": "out", "type": "reply", "payload": "x"},
            {"type": "wait"},
        ],
    }
    annotated = (
        record["system"] if target == "system" else record["items"][target == "wait"]
    )
    annotated["loss_weight"] = weight
    with pytest.raises(ValidationError, match="requires native schema_version 2"):
        Conversation.from_json(json.dumps(record))


def test_v1_schema_property_named_loss_weight_is_not_training_metadata():
    record = Conversation(
        system=SystemPrompt.build(
            tools=[
                ToolSpec(
                    name="t",
                    description="",
                    json_schema={
                        "type": "object",
                        "properties": {"loss_weight": {"type": "number"}},
                    },
                )
            ]
        )
    )
    old = json.loads(record.to_json())
    old["schema_version"] = 1
    assert Conversation.from_json(json.dumps(old)) == record


def test_weighted_factories_and_all_conversation_shortcuts():
    call = Call.build(name="t", counter=0, arguments={}, loss_weight=0.5)
    assert (
        Call.builder().name("t").counter(0).arguments({}).loss_weight(0.5).build()
        == call
    )
    assert SystemPrompt.builder().loss_weight(0.5).build() == SystemPrompt.build(
        loss_weight=0.5
    )
    record = (
        Conversation.builder()
        .host("host", loss_weight=0.5)
        .user("user", loss_weight=0.5)
        .document("doc", from_="user", id="d", loss_weight=0.5)
        .result(name="t", counter=0, payload="result", loss_weight=0.5)
        .call(name="t", counter=0, arguments={}, loss_weight=0.5)
        .think("think", loss_weight=0.5)
        .reply("reply", loss_weight=0.5)
        .claim("claim", loss_weight=0.5)
        .wait(loss_weight=0.5)
        .build()
    )
    assert all(item.loss_weight == 0.5 for item in record.items)
    assert Wait.builder().loss_weight(0).build() == Wait(loss_weight=0)
