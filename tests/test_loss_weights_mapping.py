"""OpenAI projection rejects training annotation loss unless explicitly allowed."""

import pytest

from apertus_common import (
    AdapterError,
    Call,
    Claim,
    Conversation,
    MappingPolicy,
    MessageCompleted,
    MessageStart,
    OpenAIStreamMapper,
    PayloadDelta,
    Reply,
    Result,
    SystemPrompt,
    User,
    Wait,
    from_openai,
    to_openai,
    to_openai_output,
)


@pytest.mark.parametrize(
    "target", ["system", "user", "reply", "wait", "call", "result"]
)
def test_native_history_requires_explicit_annotation_omission(target):
    record = Conversation(
        system=SystemPrompt.build(loss_weight=0 if target == "system" else None),
        items=[
            User(payload="input", loss_weight=0 if target == "user" else None),
            Reply(payload="reply", loss_weight=1 if target == "reply" else None),
            Call(
                name="t",
                counter=0,
                payload="{}",
                loss_weight=1 if target == "call" else None,
            ),
            Wait(loss_weight=1 if target == "wait" else None),
            Result(
                name="t",
                counter=0,
                payload="done",
                loss_weight=0 if target == "result" else None,
            ),
        ],
    )
    expected_index = {
        "system": -1,
        "user": 0,
        "reply": 1,
        "call": 2,
        "wait": 3,
        "result": 4,
    }[target]
    with pytest.raises(AdapterError, match="loss weights") as caught:
        to_openai(record)
    assert caught.value.index == expected_index
    result = to_openai(record, policy=MappingPolicy(omit_loss_weights=True))
    assert [item.code for item in result.diagnostics] == ["loss-weight-omitted"]
    assert result.diagnostics[0].lossy
    assert result.diagnostics[0].index == expected_index
    assert all("loss_weight" not in message for message in result.value["messages"])


@pytest.mark.parametrize("target", ["reply", "wait"])
def test_annotation_error_uses_original_index_in_later_assistant_run(target):
    record = Conversation(
        items=[
            User(payload="first"),
            Reply(payload="answer"),
            Wait(),
            User(payload="second"),
            Reply(payload="answer", loss_weight=0.5 if target == "reply" else None),
            Wait(loss_weight=0.5 if target == "wait" else None),
        ]
    )
    with pytest.raises(AdapterError, match="loss weights") as caught:
        to_openai(record)
    assert caught.value.where == "item"
    assert caught.value.index == (4 if target == "reply" else 5)


@pytest.mark.parametrize("target", ["reply", "wait"])
def test_output_projection_checks_annotation_before_it_is_lost(target):
    output = Conversation(
        items=[
            Reply(payload="answer", loss_weight=1 if target == "reply" else None),
            Wait(loss_weight=1 if target == "wait" else None),
        ]
    )
    with pytest.raises(AdapterError, match="loss weights"):
        to_openai_output(output)
    result = to_openai_output(output, policy=MappingPolicy(omit_loss_weights=True))
    assert result.value == {"role": "assistant", "content": "answer"}
    assert [item.code for item in result.diagnostics] == ["loss-weight-omitted"]


def test_bare_wait_requires_weight_policy_as_well_as_boundary_policy():
    record = Conversation(items=[Wait(loss_weight=0)])
    with pytest.raises(AdapterError, match="loss weights"):
        to_openai(record, policy=MappingPolicy(merge_assistant_messages=True))
    result = to_openai(
        record,
        policy=MappingPolicy(merge_assistant_messages=True, omit_loss_weights=True),
    )
    assert {item.code for item in result.diagnostics} == {
        "loss-weight-omitted",
        "wait-omitted",
    }


def test_stream_annotation_failure_is_terminal_when_completion_reveals_it():
    mapper = OpenAIStreamMapper()
    mapper.push(MessageStart(0, "reply", {}))
    assert mapper.push(PayloadDelta(0, "answer")) == [{"content": "answer"}]
    with pytest.raises(AdapterError, match="loss weights"):
        mapper.push(MessageCompleted(0, Reply(payload="answer", loss_weight=1)))
    with pytest.raises(AdapterError):
        mapper.finish()


def test_stream_and_output_omission_diagnostics_agree():
    message = Claim(payload="42", loss_weight=0.5)
    policy = MappingPolicy(omit_claims=True, omit_loss_weights=True)
    mapper = OpenAIStreamMapper(policy=policy)
    mapper.push(MessageStart(0, "claim", {}))
    mapper.push(PayloadDelta(0, "42"))
    mapper.push(MessageCompleted(0, message))
    result = mapper.finish()
    assert result == to_openai_output(Conversation(items=[message]), policy=policy)
    assert [item.code for item in result.diagnostics] == [
        "claim-omitted",
        "loss-weight-omitted",
    ]
    with pytest.raises(AdapterError, match="unsupported output type"):
        to_openai_output(
            Conversation(items=[message]), policy=MappingPolicy(omit_loss_weights=True)
        )


@pytest.mark.parametrize("as_model", [False, True])
def test_native_system_defaults_preserve_annotations_when_mapping_from_openai(as_model):
    defaults = SystemPrompt.build(loss_weight=0.5) if as_model else {"loss_weight": 0.5}
    result = from_openai([{"role": "user", "content": "hi"}], system_defaults=defaults)
    assert result.value.system.loss_weight == 0.5
    assert result.diagnostics == ()
    assert (
        from_openai([], system_defaults=SystemPrompt.build()).value.system.loss_weight
        is None
    )
