"""Public surface: exports and schema export."""

from jsonschema import Draft202012Validator

import apertus_common
from apertus_common import Conversation
from conftest import full_cast

EXPECTED = {
    "AdapterError",
    "ApertusFormatError",
    "Assistant",
    "Attachment",
    "AudioPart",
    "Checker",
    "ConformanceError",
    "ConformanceReport",
    "Conversation",
    "Document",
    "Effort",
    "Event",
    "FramingError",
    "GenericInput",
    "GenericOutput",
    "Harness",
    "HeaderError",
    "IdConventionChecker",
    "ImagePart",
    "IncompleteMessageError",
    "InputMessage",
    "InputOrderChecker",
    "Location",
    "MessageValidationError",
    "OutputMessage",
    "ParseFailure",
    "ParserStateError",
    "PayloadDelta",
    "Retrieval",
    "Rule",
    "Severity",
    "StreamParseError",
    "StreamParser",
    "SystemPrompt",
    "SystemPromptChecker",
    "TextPart",
    "Think",
    "ToolCall",
    "ToolCallChecker",
    "ToolResult",
    "ToolResultChecker",
    "ToolSpec",
    "TypeInfo",
    "User",
    "VerifiableAnswer",
    "VerifiableAnswerChecker",
    "Violation",
    "Wait",
    "WaitChecker",
    "check",
    "default_checkers",
    "delivery_order",
    "from_json",
    "from_openai",
    "is_canonical_id",
    "render_str",
    "rule_table",
    "to_json",
    "to_openai",
    "type_info",
}


def test_public_surface_matches_spec() -> None:
    assert set(apertus_common.__all__) == EXPECTED
    for name in EXPECTED:
        assert hasattr(apertus_common, name)


def test_json_schema_encodes_runtime_constraints() -> None:
    schema = Conversation.model_json_schema()
    definitions = schema["$defs"]

    generic_input_type = definitions["GenericInput"]["properties"]["type"]
    assert "harness" in generic_input_type["not"]["enum"]
    generic_output_type = definitions["GenericOutput"]["properties"]["type"]
    assert "assistant" in generic_output_type["not"]["enum"]

    tool_result = definitions["ToolResult"]
    assert "pattern" in tool_result["properties"]["id"]
    extras = tool_result["properties"]["extras"]
    assert "pattern" in extras["propertyNames"]
    assert "id" in extras["propertyNames"]["not"]["enum"]

    assert "oneOf" in definitions["ImagePart"]


def test_json_schema_accepts_dumps_and_rejects_invalid_models() -> None:
    schema = Conversation.model_json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(full_cast().model_dump(mode="json"))

    invalid_models = [
        {"items": [{"id": "f:1", "payload": "x"}]},
        {
            "items": [
                {
                    "direction": "in",
                    "type": "user",
                    "payload": [{"text": "x"}],
                }
            ]
        },
        {"items": [{}]},
        {
            "items": [
                {
                    "direction": "out",
                    "type": "tool_call",
                    "id": "f:1",
                    "name": "f",
                    "args": {"x": "\ud800"},
                }
            ]
        },
        {
            "items": [
                {
                    "direction": "out",
                    "type": "tool_call",
                    "id": "f:1",
                    "name": "f",
                    "args": {"\ud800": "x"},
                }
            ]
        },
        {
            "items": [
                {
                    "direction": "in",
                    "type": "harness",
                    "payload": "x",
                    "extras": {"bad key": "x"},
                }
            ]
        },
        {
            "items": [
                {
                    "direction": "in",
                    "type": "tool_result",
                    "id": "bad id",
                    "payload": "x",
                }
            ]
        },
        {
            "items": [
                {
                    "direction": "in",
                    "type": "user",
                    "payload": [{"part": "image", "height": 2}],
                }
            ]
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
