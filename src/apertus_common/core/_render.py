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
    payload_weight = message.loss_weight
    if payload_weight is None:
        payload_weight = 0.0 if is_input else 1.0
    frame_weight = 0.0 if is_input else payload_weight
    yield Control(ControlToken.IN if is_input else ControlToken.OUT, frame_weight)
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
    yield Text(
        message.type + (_json(headers) if headers else ""), "structural", frame_weight
    )
    yield Control(ControlToken.HDR, frame_weight)
    if message.payload:
        yield Text(message.payload, "payload", payload_weight)
    yield Control(
        ControlToken.IN_END if is_input else ControlToken.OUT_END, frame_weight
    )


def to_wire(conversation: Conversation) -> tuple[WireAtom, ...]:
    """Emit envelopes without synthetic whitespace, waits, or sorting."""
    atoms: list[WireAtom] = []
    if conversation.system is not None:
        # `tools` is always present; an empty array means no tools.
        system = conversation.system.model_dump(
            mode="json", by_alias=True, exclude_none=True, exclude={"loss_weight"}
        )
        atoms.extend(
            (
                Control(ControlToken.IN),
                Text("system", "structural"),
                Control(ControlToken.HDR),
                Text(_json(system), "payload", conversation.system.loss_weight or 0.0),
                Control(ControlToken.IN_END),
            )
        )
    for item in conversation.items:
        if isinstance(item, Wait):
            weight = 1.0 if item.loss_weight is None else item.loss_weight
            atoms.append(Control(ControlToken.WAIT, weight))
        else:
            atoms.extend(_message_atoms(item))
    return tuple(atoms)


def render_str(conversation: Conversation) -> str:
    """Diagnostic glyph projection; literal control spellings are ambiguous."""
    return "".join(
        atom.token.value if isinstance(atom, Control) else atom.text
        for atom in to_wire(conversation)
    )
