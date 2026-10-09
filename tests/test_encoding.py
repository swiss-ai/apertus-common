"""Synthetic ByteLevel unit artifacts; production artifact checks are separate."""

import json
from unittest.mock import Mock

import pytest
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from apertus_common import (
    Call,
    Claim,
    Conversation,
    Document,
    EncodingResult,
    Host,
    MessageCompleted,
    ParseBoundary,
    ParseFailure,
    ParserStateError,
    PayloadDelta,
    Reply,
    Result,
    StreamParseError,
    SystemPrompt,
    Think,
    TokenizerError,
    UnknownOutput,
    User,
    Wait,
    load_encoding,
    render_str,
)

CONTROLS = [
    "<|in|>",
    "<|/in|>",
    "<|out|>",
    "<|/out|>",
    "<|hdr|>",
    "<|wait|>",
    "<|pad|>",
]
BOS = "<s>"
ROLES = dict(
    zip(
        (
            "input_start_token",
            "input_end_token",
            "output_start_token",
            "output_end_token",
            "header_end_token",
            "wait_token",
        ),
        CONTROLS[:-1],
        strict=True,
    )
)


def write_state(directory, state):
    (directory / "tokenizer.json").write_text(json.dumps(state, ensure_ascii=False))


@pytest.fixture
def artifact(tmp_path):
    specials = [*CONTROLS, "<|other|>", BOS]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: index for index, token in enumerate([*specials, *alphabet, "l{"])}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[("l", "{")]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [AddedToken(token, normalized=False, special=True) for token in specials]
    )
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        extra_special_tokens=ROLES,
        pad_token=CONTROLS[-1],
        bos_token=BOS,
        clean_up_tokenization_spaces=False,
    )
    hf.save_pretrained(tmp_path)
    return tmp_path


@pytest.fixture
def encoding(artifact):
    return load_encoding(artifact)


def test_ordinary_text_is_verbatim_and_never_special(encoding):
    for payload in ["", *CONTROLS, "<|other|>", " e\u0301\r\n\x00 🙂漢字\ufffd "]:
        ids = encoding.encode_text(payload)
        assert encoding.special_token_ids.isdisjoint(ids)
        assert encoding.decode_text(ids) == payload


@pytest.mark.parametrize("legacy_merges", [False, True])
def test_bpe_merge_cannot_produce_special_id(artifact, legacy_merges):
    state = json.loads((artifact / "tokenizer.json").read_text())
    token_id = state["model"]["vocab"]["l{"]
    state["added_tokens"].append(
        {
            "id": token_id,
            "content": "l{",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    if legacy_merges:
        state["model"]["merges"] = ["l {"]
    write_state(artifact, state)
    with pytest.raises(TokenizerError, match="BPE merges must not produce special"):
        load_encoding(artifact)


def test_controls_and_generation_constraints(encoding):
    controls = encoding.controls
    allowed = {
        controls.output_start,
        controls.output_end,
        controls.header_end,
        controls.wait,
    }
    assert encoding.generation_constraints.eos_token_id == controls.wait
    assert (
        set(encoding.generation_constraints.forbidden_token_ids)
        == encoding.special_token_ids - allowed
    )
    assert encoding.get_vocab()["<|wait|>"] == controls.wait
    assert encoding.id_to_token(controls.wait) == "<|wait|>"


def test_roundtrip_reconstructs_without_running_checks(encoding):
    expected = Conversation(
        items=[
            Think(payload="résumé🙂 e\u0301"),
            Call(name="unknown", counter=-1, payload="not JSON <|wait|>"),
            UnknownOutput(output_type="extension", headers={"x": "v"}, payload="\x00"),
            Claim(payload="value"),
            Reply(payload="literal <|out|>reply<|hdr|> text <|/out|>"),
            Wait(),
        ]
    )
    ids = encoding.encode_items(iter(expected.items))
    assert encoding.parse_output(ids) == expected
    assert encoding.encode_conversation(expected) == [encoding.bos_token_id, *ids]
    for chunk_size in [1, 2, 7, len(ids)]:
        parser = encoding.output_parser()
        events = []
        for offset in range(0, len(ids), chunk_size):
            parser.push_tokens(ids[offset : offset + chunk_size])
            events.extend(parser.take_events())
        assert parser.finish() == expected
        assert [
            event.message for event in events if isinstance(event, MessageCompleted)
        ] == list(expected.items[:-1])
        assert parser.boundary is ParseBoundary.WAIT
        assert parser.consumed == len(ids)
        assert parser.last_valid_boundary == len(ids)


def test_conversation_starts_with_bos_and_items_never_add_it(encoding):
    conversation = Conversation(
        system=SystemPrompt.build(), items=[Reply(payload="<s> stays text")]
    )
    ids = encoding.encode_conversation(conversation)
    assert ids[0] == encoding.bos_token_id == encoding.token_to_id(BOS)
    assert ids.count(encoding.bos_token_id) == 1
    assert encoding.decode_text(ids[1:]) == render_str(conversation)
    assert encoding.bos_token_id not in encoding.encode_items(conversation.items)
    assert encoding.bos_token_id in encoding.generation_constraints.forbidden_token_ids


@pytest.mark.parametrize("value", [None, "<|in|>", "<not-registered>"])
def test_missing_or_control_bos_rejected(artifact, value):
    tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
    tokenizer.bos_token = value
    with pytest.raises(TokenizerError, match="bos_token"):
        load_encoding(tokenizer)


def test_header_text_is_encoded_as_one_run(encoding):
    call = Call(name="x", counter=0, payload="{}")
    ids = encoding.encode_items([call])
    end = ids.index(encoding.controls.header_end)
    assert ids[1:end] == encoding.encode_text('call{"name":"x","counter":0}')
    assert encoding.token_to_id("l{") in ids[1:end]


def test_empty_call_name_fails_at_header_token_without_events(encoding):
    parser = encoding.output_parser()
    parser.push_tokens(
        [
            encoding.controls.output_start,
            *encoding.encode_text('call{"name":"","counter":0}'),
        ]
    )
    with pytest.raises(StreamParseError) as caught:
        parser.push_tokens([encoding.controls.header_end])
    assert caught.value.reason is ParseFailure.INVALID_HEADER
    assert parser.current is None
    assert parser.take_events() == []


def test_unicode_split_emits_no_replacement_text(encoding):
    parser = encoding.output_parser()
    prefix = [
        encoding.controls.output_start,
        *encoding.encode_text("reply"),
        encoding.controls.header_end,
    ]
    parser.push_tokens(prefix)
    parser.take_events()
    for token in encoding.encode_text("🙂"):
        parser.push_tokens([token])
        assert all(
            "\ufffd" not in event.text
            for event in parser.take_events()
            if isinstance(event, PayloadDelta)
        )
    assert parser.current.payload == "🙂"
    parser.push_tokens([encoding.controls.output_end])
    assert parser.finish() == Conversation(items=[Reply(payload="🙂")])


def test_clean_finish_is_idempotent_and_terminal(encoding):
    parser = encoding.output_parser()
    parser.push_tokens(encoding.encode_items([Reply(payload="done")]))
    first = parser.finish()
    assert parser.finish() is first
    assert parser.boundary is ParseBoundary.MESSAGE_END
    with pytest.raises(ParserStateError):
        parser.push_tokens([])
    empty = encoding.output_parser()
    assert empty.finish() == Conversation()
    assert empty.boundary is ParseBoundary.EMPTY


@pytest.mark.parametrize("tail", ["normal", "invalid-byte", "invalid-id", "special"])
def test_wait_rejects_all_trailing_data_before_decoding(encoding, tail):
    token = {
        "normal": encoding.token_to_id("a"),
        "invalid-byte": encoding.token_to_id("ÿ"),
        "invalid-id": -1,
        "special": encoding.controls.input_start,
    }[tail]
    parser = encoding.output_parser()
    with pytest.raises(StreamParseError) as caught:
        parser.push_tokens([encoding.controls.wait, token])
    assert caught.value.reason is ParseFailure.TRAILING_AFTER_WAIT
    assert caught.value.offset == caught.value.last_valid_boundary == 1
    assert caught.value.completed == Conversation(items=[Wait()])
    with pytest.raises(ParserStateError):
        parser.finish()
    with pytest.raises(ParserStateError):
        parser.push_tokens([])


def test_wait_finish_and_later_push(encoding):
    parser = encoding.output_parser()
    parser.push_tokens([encoding.controls.wait])
    with pytest.raises(ParserStateError):
        parser.push_tokens([])
    result = parser.finish()
    assert result == Conversation(items=[Wait()])
    assert parser.finish() is result


@pytest.mark.parametrize("close", [False, True])
def test_incomplete_unicode_is_terminal_with_safe_recovery(encoding, close):
    completed = Conversation(items=[Reply(payload="safe")])
    original = encoding.encode_items(completed.items)
    parser = encoding.output_parser()
    parser.push_tokens(original)
    parser.push_tokens(
        [
            encoding.controls.output_start,
            *encoding.encode_text("reply"),
            encoding.controls.header_end,
            encoding.token_to_id("Ã"),
        ]
    )
    with pytest.raises(StreamParseError) as caught:
        if close:
            parser.push_tokens([encoding.controls.output_end])
        else:
            parser.finish()
    assert caught.value.reason is ParseFailure.INVALID_ENCODING
    assert caught.value.completed == completed
    assert caught.value.last_valid_boundary == len(original)
    assert caught.value.unit == "token"
    with pytest.raises(ParserStateError):
        parser.finish()


@pytest.mark.parametrize("token", [-1, True, 1.0, 999999999])
def test_invalid_ids_have_typed_errors(encoding, token):
    with pytest.raises(TokenizerError):
        encoding.decode_text([token])
    assert encoding.id_to_token(token) is None
    with pytest.raises(StreamParseError) as caught:
        encoding.parse_output([token])
    assert caught.value.reason is ParseFailure.INVALID_ENCODING


def test_forbidden_noncontrol_special_fails(encoding):
    with pytest.raises(StreamParseError) as caught:
        encoding.parse_output([encoding.token_to_id("<|other|>")])
    assert caught.value.reason is ParseFailure.UNEXPECTED_CONTROL


@pytest.mark.parametrize("payload", ["\ud800", None])
def test_invalid_ordinary_text_fails(encoding, payload):
    with pytest.raises(TokenizerError):
        encoding.encode_text(payload)


@pytest.mark.parametrize(
    "damage",
    [
        "remove-text",
        "duplicate-id",
        "ordinary-added",
        "prefix-space",
        "normalizer",
        "missing-byte",
    ],
)
def test_artifact_rejects_unsafe_backend(artifact, damage):
    state = json.loads((artifact / "tokenizer.json").read_bytes())
    if damage == "remove-text":
        state["pre_tokenizer"] = {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"String": "Z"},
                    "behavior": "Removed",
                    "invert": False,
                },
                state["pre_tokenizer"],
            ],
        }
    elif damage == "duplicate-id":
        state["model"]["vocab"]["b"] = state["model"]["vocab"]["a"]
    elif damage == "ordinary-added":
        state["added_tokens"][7]["special"] = False
    elif damage == "prefix-space":
        state["pre_tokenizer"]["add_prefix_space"] = True
    elif damage == "normalizer":
        state["normalizer"] = {"type": "Lowercase"}
    elif damage == "missing-byte":
        token_id = state["model"]["vocab"].pop("Z")
        state["model"]["vocab"]["ZZ"] = token_id
    write_state(artifact, state)
    with pytest.raises(TokenizerError):
        load_encoding(artifact)


@pytest.mark.parametrize("role", [*ROLES, "pad_token"])
@pytest.mark.parametrize("value", [None, "<|other|>"])
def test_missing_or_wrong_control_role_rejected(artifact, role, value):
    tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
    setattr(tokenizer, role, value)
    with pytest.raises(TokenizerError, match=role):
        load_encoding(tokenizer)


def test_binding_snapshots_caller_backend(artifact):
    tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
    backend = tokenizer.backend_tokenizer
    backend.enable_padding(length=100, pad_id=6, pad_token="<|pad|>")
    backend.enable_truncation(max_length=3)
    backend.post_processor = processors.TemplateProcessing(
        single="<|in|> $A",
        special_tokens=[("<|in|>", 0)],
    )
    before = backend.to_str()
    encoding = load_encoding(tokenizer)
    payload = "literal <|out|> and <|other|> e\u0301"
    ids = encoding.encode_text(payload)
    assert encoding.decode_text(ids) == payload
    assert encoding.special_token_ids.isdisjoint(ids)
    assert backend.to_str() == before
    assert not backend.encode_special_tokens
    assert 7 in encoding.special_token_ids
    assert 7 not in tokenizer.all_special_ids  # unnamed specials still excluded
    backend.add_special_tokens(["<|later|>"])
    tokenizer.output_start_token = "<|other|>"
    assert encoding.encode_text(payload) == ids
    assert encoding.controls.output_start == 2
    assert encoding.token_to_id("<|later|>") is None


@pytest.mark.parametrize("value", [None, object(), 1])
def test_invalid_tokenizer_object_rejected(value):
    with pytest.raises(TokenizerError):
        load_encoding(value)


def test_missing_or_malformed_artifact_rejected(tmp_path):
    with pytest.raises(TokenizerError):
        load_encoding(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{")
    with pytest.raises(TokenizerError):
        load_encoding(tmp_path)


def test_added_only_specials_extend_the_vocabulary(artifact):
    state = json.loads((artifact / "tokenizer.json").read_bytes())
    token_id = len(state["model"]["vocab"])
    state["added_tokens"].append(
        {
            "id": token_id,
            "content": "<|added-only|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    write_state(artifact, state)
    encoding = load_encoding(artifact)
    assert encoding.token_to_id("<|added-only|>") == token_id
    assert token_id in encoding.generation_constraints.forbidden_token_ids
    assert (
        encoding.decode_text(encoding.encode_text("<|added-only|>")) == "<|added-only|>"
    )


@pytest.mark.parametrize("weight", [None, 0.0, 0.25, 2.0])
@pytest.mark.parametrize(
    "message",
    [
        User(payload="hé🙂 <|out|>"),
        Host(payload="hé🙂 <|out|>"),
        Document(from_="user", id="d", payload="hé🙂 <|out|>"),
        Result(name="f", counter=0, payload="hé🙂 <|out|>"),
        Think(payload="hé🙂 <|out|>"),
        Reply(payload="hé🙂 <|out|>"),
        Claim(payload="hé🙂 <|out|>"),
        Call(name="f", counter=-1, payload="hé🙂 <|out|>"),
        UnknownOutput(output_type="other", headers={"x": "y"}, payload="hé🙂 <|out|>"),
    ],
)
def test_loss_weights_follow_message_regions_without_changing_tokens(
    encoding, message, weight
):
    weighted = type(message).model_validate(
        {**message.model_dump(), "loss_weight": weight}
    )
    result = encoding.encode_items([weighted], return_loss_weights=True)
    assert isinstance(result, EncodingResult)
    assert result.token_ids == encoding.encode_items([message])
    assert result.token_ids == encoding.encode_items([weighted])
    assert len(result.loss_weights) == len(result.token_ids)
    is_input = message.direction == "in"
    payload_weight = weight if weight is not None else (0.0 if is_input else 1.0)
    frame_weight = 0.0 if is_input else payload_weight
    delimiter = result.token_ids.index(encoding.controls.header_end)
    assert result.loss_weights[: delimiter + 1] == [frame_weight] * (delimiter + 1)
    assert result.loss_weights[delimiter + 1 : -1] == [payload_weight] * len(
        encoding.encode_text(message.payload)
    )
    assert result.loss_weights[-1] == frame_weight


def test_system_weight_never_enters_wire_and_bos_is_zero(encoding):
    plain = Conversation(system=SystemPrompt.build(behavior="Hello"))
    weighted = Conversation(
        system=SystemPrompt(thinking="medium", behavior="Hello", loss_weight=0.5)
    )
    result = encoding.encode_conversation(weighted, return_loss_weights=True)
    assert result.token_ids == encoding.encode_conversation(plain)
    assert render_str(weighted) == render_str(plain)
    assert result.loss_weights[:2] == [0.0, 0.0]
    delimiter = result.token_ids.index(encoding.controls.header_end)
    assert result.loss_weights[: delimiter + 1] == [0.0] * (delimiter + 1)
    assert result.loss_weights[delimiter + 1 : -1] == [0.5] * (
        len(result.token_ids) - delimiter - 2
    )
    assert result.loss_weights[-1] == 0.0


def test_empty_payloads_and_wait_have_independent_weights(encoding):
    items = [Reply(payload="", loss_weight=0.0), Wait(), Wait(loss_weight=0.25)]
    result = encoding.encode_items(iter(items), return_loss_weights=True)
    assert result.loss_weights == [0.0] * (len(result.token_ids) - 2) + [1.0, 0.25]
    assert encoding.encode_items([], return_loss_weights=True) == EncodingResult([], [])
    assert encoding.encode_conversation(
        Conversation(), return_loss_weights=True
    ) == EncodingResult([encoding.bos_token_id], [0.0])


def test_scalar_weights_use_one_tokenization_per_text_run(encoding):
    backend = Mock(wraps=encoding._backend)
    encoding._backend = backend
    result = encoding.encode_items(
        [Call(name="f", counter=0, payload="{}", loss_weight=0.5)],
        return_loss_weights=True,
    )
    assert backend.encode.call_count == 2
    assert backend.encode_batch_fast.call_count == 0
    assert result.loss_weights == [0.5] * len(result.token_ids)


def test_batch_matches_scalar_weighted_and_unweighted_with_single_backend_batch(
    encoding,
):
    conversations = [
        Conversation(),
        Conversation(system=SystemPrompt(thinking="high", loss_weight=0.5)),
        Conversation(
            items=[User(payload="x", loss_weight=2), Reply(payload="🙂 <|wait|>")]
        ),
        Conversation(
            items=[
                Call(name="f", counter=-1, payload="not JSON", loss_weight=0.25),
                Wait(),
            ]
        ),
        Conversation(items=[Wait(loss_weight=0)]),
    ]
    scalar_ids = [encoding.encode_conversation(value) for value in conversations]
    scalar_weighted = [
        encoding.encode_conversation(value, return_loss_weights=True)
        for value in conversations
    ]
    backend = Mock(wraps=encoding._backend)
    encoding._backend = backend
    assert encoding.encode_conversations(conversations) == scalar_ids
    assert (
        encoding.encode_conversations(conversations, return_loss_weights=True)
        == scalar_weighted
    )
    assert backend.encode.call_count == 0
    assert backend.encode_batch_fast.call_count == 2
    assert (
        'call{"name":"f","counter":-1}' in backend.encode_batch_fast.call_args.args[0]
    )
    assert encoding.encode_conversations([]) == []
    assert encoding.encode_conversations([], return_loss_weights=True) == []
    assert encoding.encode_conversations(
        [Conversation()], return_loss_weights=True
    ) == [EncodingResult([encoding.bos_token_id], [0.0])]
    assert backend.encode_batch_fast.call_count == 2


@pytest.mark.parametrize("name", ["memo", "user", "wait", "unknown_output"])
def test_unknown_output_storage_tag_never_enters_wire(encoding, name):
    message = UnknownOutput(output_type=name, payload="hé🙂 <|out|>")
    items = [message, Wait()]
    expected = [
        encoding.controls.output_start,
        *encoding.encode_text(name),
        encoding.controls.header_end,
        *encoding.encode_text(message.payload),
        encoding.controls.output_end,
        encoding.controls.wait,
    ]
    assert encoding.encode_items(items) == expected
    assert encoding.parse_output(expected) == Conversation(items=items)
    parser = encoding.output_parser()
    for token in expected:
        parser.push_tokens([token])
    assert parser.finish() == Conversation(items=items)
