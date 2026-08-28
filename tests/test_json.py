"""formats: from_json/to_json and the ingest flags (normalize, check, strict)."""

import pickle

import pytest
from pydantic import ValidationError

from apertus_common import (
    Assistant,
    ConformanceError,
    ConformanceReport,
    Conversation,
    Harness,
    User,
    Wait,
    from_json,
    to_json,
)
from conftest import full_cast, pretraining

MESSY = Conversation(items=[User(payload="hi"), Harness(payload="notice")])


def test_aliases_match_the_model_json() -> None:
    for conv in (full_cast(), pretraining()):
        assert to_json(conv) == conv.model_dump_json()
        assert from_json(to_json(conv), normalize=False) == conv
    with pytest.raises(ValidationError):
        from_json('{"items": [{"direction": "in", "type": "user"}]}')
    with pytest.raises(ValidationError):
        from_json("not json")


def test_ingest_flags() -> None:
    text = to_json(MESSY)
    assert from_json(text) == MESSY.normalized()
    assert from_json(text, normalize=False) == MESSY

    conv, report = from_json(text, normalize=False, check=True)
    assert conv == MESSY and isinstance(report, ConformanceReport) and not report.ok
    conv, report = from_json(text, check=True)
    assert conv == MESSY.normalized() and report.ok

    with pytest.raises(ConformanceError) as info:
        from_json(text, normalize=False, strict=True)
    error = info.value
    assert error.report.errors[0].rule == "input-order/canonical-order"
    assert str(error).startswith("1 conformance error(s): input-order/canonical-order")
    assert pickle.loads(pickle.dumps(error)).report == error.report
    assert from_json(text, strict=True) == MESSY.normalized()  # normalized first

    # warnings alone never raise: a trailing open output run is a warning
    open_run = Conversation(
        items=[
            User(payload="a"),
            Assistant(payload="b"),
            Wait(),
            Assistant(payload="c"),
        ]
    )
    conv, report = from_json(to_json(open_run), check=True, strict=True)
    assert report.ok and report.warnings and conv == open_run


def test_json_lines_recipe() -> None:
    dataset = [full_cast(), pretraining(), MESSY]
    lines = "".join(to_json(conv) + "\n" for conv in dataset)
    assert [from_json(line, normalize=False) for line in lines.splitlines()] == dataset
