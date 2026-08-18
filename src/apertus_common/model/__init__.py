"""Public typed model for constructing and inspecting Apertus conversations.

This package collects the concrete classes implemented across the private
``_base``, ``_primitives``, ``_system``, ``_input``, ``_output``, and
``_conversation`` modules. Import from here or, preferably, from the flat
top-level :mod:`apertus_common` API; the private module layout is not a public
compatibility contract.
"""

from apertus_common.model._base import (
    InputMessage,
    OutputMessage,
    TypeInfo,
    type_info,
)
from apertus_common.model._conversation import (
    Conversation,
    Wait,
    delivery_order,
)
from apertus_common.model._input import (
    Attachment,
    Document,
    Event,
    GenericInput,
    Harness,
    Retrieval,
    ToolResult,
    User,
)
from apertus_common.model._output import (
    Assistant,
    GenericOutput,
    Think,
    ToolCall,
    VerifiableAnswer,
)
from apertus_common.model._primitives import (
    AudioPart,
    Effort,
    ImagePart,
    TextPart,
    is_canonical_id,
)
from apertus_common.model._system import SystemPrompt, ToolSpec

__all__ = [
    "Assistant",
    "Attachment",
    "AudioPart",
    "Conversation",
    "Document",
    "Effort",
    "Event",
    "GenericInput",
    "GenericOutput",
    "Harness",
    "ImagePart",
    "InputMessage",
    "OutputMessage",
    "Retrieval",
    "SystemPrompt",
    "TextPart",
    "Think",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "TypeInfo",
    "User",
    "VerifiableAnswer",
    "Wait",
    "delivery_order",
    "is_canonical_id",
    "type_info",
]
