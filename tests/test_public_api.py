"""Public surface: exports and schema export."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

import apertus_common
from apertus_common import Conversation
from conftest import full_cast

EXPECTED = {
    "ConformanceError",
    "Think",
    "ConversationChecker",
    "ParseBoundary",
    "Thinking",
    "to_openai",
    "ParseFailure",
    "MappingDiagnostic",
    "Document",
    "OpenAIStreamMapper",
    "rule_table",
    "ConformanceReport",
    "Rule",
    "CallState",
    "AdapterError",
    "Call",
    "CheckOptions",
    "ArgumentError",
    "User",
    "to_openai_output",
    "HostClosure",
    "HistoryCheckpoint",
    "Conversation",
    "InputMessage",
    "MappingResult",
    "Claim",
    "OpenAIMappingContext",
    "OutputMessage",
    "parse_output_text",
    "JsonObject",
    "PayloadDelta",
    "load_encoding",
    "CheckMode",
    "OpenAIMessage",
    "ProfileChecker",
    "Checker",
    "ToolSpec",
    "MappingPolicy",
    "GenerationConstraints",
    "Reply",
    "OpenMessage",
    "Message",
    "Severity",
    "Violation",
    "from_openai",
    "TokenStreamParser",
    "SystemPrompt",
    "OpenAIRequest",
    "Unevaluated",
    "parse_arguments",
    "render_str",
    "Location",
    "ArgumentsChecker",
    "Result",
    "check_arguments",
    "ApertusFormatError",
    "MessageCompleted",
    "OpenAIAssistantMessage",
    "checkers_for",
    "Host",
    "ParserStateError",
    "CheckContext",
    "TokenizerError",
    "ParseEvent",
    "StreamParseError",
    "UnknownOutput",
    "ApertusEncoding",
    "MessageStart",
    "OpenAIDelta",
    "WaitReached",
    "ControlTokenIds",
    "TextStreamParser",
    "Wait",
    "ToolAssociation",
}


def test_public_surface_matches_spec() -> None:
    assert set(apertus_common.__all__) == EXPECTED
    for name in EXPECTED:
        assert hasattr(apertus_common, name)


def test_json_schema_encodes_runtime_constraints() -> None:
    schema = Conversation.model_json_schema()
    definitions = schema["$defs"]

    assert schema["$id"] == "urn:apertus-common:native:1"
    assert "schema_version" in schema["required"]
    assert "reply" in definitions["UnknownOutput"]["properties"]["type"]["not"]["enum"]
    assert {"direction", "type", "counter", "name", "payload"} <= set(
        definitions["Call"]["required"]
    )
    assert "ImagePart" not in definitions
    assert "AudioPart" not in definitions


def test_json_schema_accepts_dumps_and_rejects_invalid_models() -> None:
    schema = Conversation.model_json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(full_cast().model_dump(mode="json", by_alias=True))

    invalid_models = [
        {"items": []},
        {"schema_version": 2, "items": []},
        {"schema_version": 1, "items": [{"type": "reply", "payload": "x"}]},
        {
            "schema_version": 1,
            "items": [
                {
                    "direction": "out",
                    "type": "call",
                    "name": "f",
                    "counter": True,
                    "payload": "{}",
                }
            ],
        },
        {
            "schema_version": 1,
            "items": [
                {
                    "direction": "in",
                    "type": "user",
                    "payload": [{"part": "image", "ref": "asset://example"}],
                }
            ],
        },
    ]

    assert all(not validator.is_valid(model) for model in invalid_models)


def test_json_schema_dynamic_anchors_are_unique() -> None:
    anchors: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            anchor = value.get("$dynamicAnchor")
            if isinstance(anchor, str):
                anchors.append(anchor)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(Conversation.model_json_schema())
    assert len(anchors) == len(set(anchors))


@pytest.mark.parametrize(
    "item",
    [
        {
            "direction": "out",
            "type": "call",
            "name": "two words",
            "counter": 0,
            "payload": "",
        },
        {
            "direction": "out",
            "type": "call",
            "name": "trailing\n",
            "counter": 0,
            "payload": "",
        },
        {"direction": "out", "type": "two words", "payload": ""},
        {"direction": "out", "type": "trailing\n", "payload": ""},
        {
            "direction": "out",
            "type": "memo",
            "headers": {"key": [{"nested": "two words"}]},
            "payload": "",
        },
        {
            "direction": "out",
            "type": "memo",
            "headers": {"key": [{"bad key": True}]},
            "payload": "",
        },
        {"direction": "out", "type": "reply", "payload": "\n\ud800"},
    ],
)
def test_exported_schema_rejects_invalid_header_and_unicode_values(
    item: dict[str, object],
) -> None:
    document = {"schema_version": 1, "items": [item]}
    with pytest.raises(ValidationError):
        Conversation.from_json(json.dumps(document))
    assert not Draft202012Validator(Conversation.model_json_schema()).is_valid(document)


def test_native_schema_snapshot() -> None:
    path = Path(__file__).parent / "fixtures" / "native-v1.schema.json"
    assert Conversation.model_json_schema() == json.loads(path.read_text())
