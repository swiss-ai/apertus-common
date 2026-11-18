"""Finite JSON accepts standard Python mappings and sequences without coercion."""

from types import MappingProxyType

import pytest
from pydantic import TypeAdapter, ValidationError

from apertus_common.model._primitives import FiniteJsonObject


def test_json_containers_accept_mapping_and_tuple_then_freeze():
    adapter = TypeAdapter(FiniteJsonObject)
    value = adapter.validate_python(MappingProxyType({"x": (1, True, None, "z")}))
    assert adapter.dump_json(value) == b'{"x":[1,true,null,"z"]}'


@pytest.mark.parametrize(
    "value",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {1: "x"},
        {"x": b"bytes"},
        {"x": {1, 2}},
        {"x": "\ud800"},
    ],
)
def test_json_invalid_values_are_not_coerced(value):
    with pytest.raises(ValidationError):
        TypeAdapter(FiniteJsonObject).validate_python(value)
