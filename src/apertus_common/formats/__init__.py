"""Explicit OpenAI compatibility mapping; native JSON lives on the models."""

from apertus_common.formats._openai import from_openai, to_openai, to_openai_output
from apertus_common.formats._stream import OpenAIStreamMapper
from apertus_common.formats._types import (
    MappingDiagnostic,
    MappingPolicy,
    MappingResult,
    OpenAIAssistantMessage,
    OpenAIDelta,
    OpenAIMappingContext,
    OpenAIMessage,
    OpenAIRequest,
    ToolAssociation,
)

__all__ = [
    "MappingDiagnostic",
    "MappingPolicy",
    "MappingResult",
    "OpenAIAssistantMessage",
    "OpenAIDelta",
    "OpenAIMappingContext",
    "OpenAIMessage",
    "OpenAIRequest",
    "OpenAIStreamMapper",
    "ToolAssociation",
    "from_openai",
    "to_openai",
    "to_openai_output",
]
