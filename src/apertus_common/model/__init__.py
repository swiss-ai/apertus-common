"""Immutable native Apertus 2 models."""

from apertus_common.model._base import InputMessage, Message, OutputMessage
from apertus_common.model._conversation import Conversation, Wait
from apertus_common.model._input import Document, Host, Result, User
from apertus_common.model._output import Call, Claim, Reply, Think, UnknownOutput
from apertus_common.model._primitives import JsonObject, Thinking
from apertus_common.model._system import SystemPrompt, ToolSpec

__all__ = [
    "Call",
    "Claim",
    "Conversation",
    "Document",
    "Host",
    "InputMessage",
    "JsonObject",
    "Message",
    "OutputMessage",
    "Reply",
    "Result",
    "SystemPrompt",
    "Think",
    "Thinking",
    "ToolSpec",
    "UnknownOutput",
    "User",
    "Wait",
]
