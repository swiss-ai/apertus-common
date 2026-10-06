"""Decoded header strings cannot contain whitespace; JSON escaping is supported."""

import json

import pytest
from pydantic import ValidationError

from apertus_common import (
    Call,
    Document,
    ParseFailure,
    Result,
    StreamParseError,
    TextStreamParser,
    ToolSpec,
    UnknownOutput,
    parse_output_text,
)


@pytest.mark.parametrize("value", ["a b", "a\tb", "a\nb", "a\u2003b", "\ud800"])
def test_headers_reject_invalid_decoded_strings(value):
    with pytest.raises(ValidationError):
        Call(name=value, counter=0, payload="")
    with pytest.raises(ValidationError):
        Document(payload="", from_=value)
    with pytest.raises(ValidationError):
        UnknownOutput(type="future", headers={"nested": [value]}, payload="")


@pytest.mark.parametrize("value", ['a"b', "a=b", "a/b", "a\\b"])
def test_header_strings_allow_json_escapable_nonwhitespace(value):
    assert Call(name=value, counter=0, payload="").name == value


@pytest.mark.parametrize("value", ["", "bad type", "type{", "type<|out|>"])
def test_unknown_type_word_grammar(value):
    with pytest.raises(ValidationError):
        UnknownOutput(type=value, payload="")


@pytest.mark.parametrize("field", ["from", "id"])
def test_document_provenance_values_are_nonempty_when_present(field):
    values = {"from": "user", "id": "a1", field: ""}
    with pytest.raises(ValidationError):
        Document(payload="", from_=values["from"], id=values["id"])
    record = {"direction": "input", "type": "document", "payload": "", **values}
    with pytest.raises(ValidationError):
        Document.from_json(json.dumps(record))


def test_tool_call_and_result_names_are_nonempty():
    with pytest.raises(ValidationError):
        ToolSpec(name="", description="", json_schema={"type": "object"})
    with pytest.raises(ValidationError):
        Call(name="", counter=0, payload="{}")
    with pytest.raises(ValidationError):
        Result(name="", counter=0, payload="")
    with pytest.raises(StreamParseError) as caught:
        parse_output_text('<|out|>call{"name":"","counter":0}<|hdr|>{}<|/out|>')
    assert caught.value.reason is ParseFailure.INVALID_HEADER


def test_empty_call_name_fails_before_emitting_provisional_events():
    parser = TextStreamParser()
    parser.push_text('<|out|>call{"name":"","counter":0}')
    with pytest.raises(StreamParseError) as caught:
        parser.push_text("<|hdr|>")
    assert caught.value.reason is ParseFailure.INVALID_HEADER
    assert parser.current is None
    assert parser.take_events() == []
