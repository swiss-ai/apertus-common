"""Immutable parser observations in source order."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from apertus_common.model import JsonObject, OutputMessage


class ParseBoundary(StrEnum):
    """The last accepted framing state."""

    EMPTY = "empty"
    MESSAGE_END = "message_end"
    WAIT = "wait"
    IN_MESSAGE = "in_message"


@dataclass(frozen=True, slots=True)
class MessageStart:
    """The header is structurally valid; payload remains provisional."""

    index: int
    type: str
    headers: JsonObject


@dataclass(frozen=True, slots=True)
class PayloadDelta:
    """A provisional payload fragment, without interpretation."""

    index: int
    text: str


@dataclass(frozen=True, slots=True)
class MessageCompleted:
    """One complete typed output message."""

    index: int
    message: OutputMessage


@dataclass(frozen=True, slots=True)
class WaitReached:
    """The output segment ended at a wait token."""


@dataclass(frozen=True, slots=True)
class OpenMessage:
    """A snapshot of the current message after its valid header."""

    index: int
    type: str
    headers: JsonObject
    payload: str


type ParseEvent = MessageStart | PayloadDelta | MessageCompleted | WaitReached
