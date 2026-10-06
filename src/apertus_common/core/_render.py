"""One canonical wire projection shared by text and token encoding."""

from __future__ import annotations

import json
from collections.abc import Iterator

from apertus_common.core._wire import Control, ControlToken, Text, WireAtom
from apertus_common.model import (
    Call,
    Conversation,
    Document,
    InputMessage,
    Message,
    Result,
    UnknownOutput,
    Wait,
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _message_atoms(message: Message) -> Iterator[WireAtom]:
    is_input = isinstance(message, InputMessage)
    yield Control(ControlToken.IN if is_input else ControlToken.OUT)
    headers: dict[str, object] = {}
    if isinstance(message, (Call, Result)):
        headers = {"name": message.name, "counter": message.counter}
    elif isinstance(message, Document):
        if message.from_ is not None:
            headers["from"] = message.from_
        if message.id is not None:
            headers["id"] = message.id
    elif isinstance(message, UnknownOutput):
        headers = message.model_dump(mode="json")["headers"]
    yield Text(message.type + (_json(headers) if headers else ""), "structural")
    yield Control(ControlToken.HDR)
    if message.payload:
        yield Text(message.payload, "payload")
    yield Control(ControlToken.IN_END if is_input else ControlToken.OUT_END)


def to_wire(conversation: Conversation) -> tuple[WireAtom, ...]:
    """Emit envelopes without synthetic whitespace, waits, or sorting."""
    atoms: list[WireAtom] = []
    if conversation.system is not None:
        system = conversation.system.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        if not system["tools"]:
            del system["tools"]
        atoms.extend(
            (
                Control(ControlToken.IN),
                Text("system", "structural"),
                Control(ControlToken.HDR),
                Text(_json(system), "payload"),
                Control(ControlToken.IN_END),
            )
        )
    for item in conversation.items:
        if isinstance(item, Wait):
            atoms.append(Control(ControlToken.WAIT))
        else:
            atoms.extend(_message_atoms(item))
    return tuple(atoms)


def render_str(conversation: Conversation) -> str:
    """Diagnostic glyph projection; literal control spellings are ambiguous."""
    return "".join(
        atom.token.value if isinstance(atom, Control) else atom.text
        for atom in to_wire(conversation)
    )
