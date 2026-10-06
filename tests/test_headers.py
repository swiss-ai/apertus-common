"""Decoded header strings cannot contain whitespace; JSON escaping is supported."""

import pytest
from pydantic import ValidationError

from apertus_common import Call, Document, UnknownOutput


@pytest.mark.parametrize("value", ["a b", "a\tb", "a\nb", "a\u2003b", "\ud800"])
def test_headers_reject_invalid_decoded_strings(value):
    with pytest.raises(ValidationError):
        Call(name=value, counter=0, payload="")
    with pytest.raises(ValidationError):
        Document(payload="", from_=value)
    with pytest.raises(ValidationError):
        UnknownOutput(type="future", headers={"nested": [value]}, payload="")


@pytest.mark.parametrize("value", ['a"b', "a=b", "a/b", "", "a\\b"])
def test_header_strings_allow_json_escapable_nonwhitespace(value):
    assert Call(name=value, counter=0, payload="").name == value


@pytest.mark.parametrize("value", ["", "bad type", "type{", "type<|out|>"])
def test_unknown_type_word_grammar(value):
    with pytest.raises(ValidationError):
        UnknownOutput(type=value, payload="")
