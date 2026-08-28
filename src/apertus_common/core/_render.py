"""Model to wire IR rendering and the string projection.

The renderer walks a validated ``Conversation`` exactly as given (never
reordering, never normalizing) and emits wire atoms; ``render_str`` joins
their string projection. All separator whitespace is emitted as explicit
structural atoms, so projections are pure concatenation.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import assert_never

from apertus_common.core._wire import Control, ControlToken, Media, Text, WireAtom
from apertus_common.model import (
    Attachment,
    AudioPart,
    Conversation,
    ImagePart,
    InputMessage,
    OutputMessage,
    Retrieval,
    SystemPrompt,
    TextPart,
    ToolCall,
    ToolResult,
    Wait,
)


def _structural(text: str) -> Text:
    return Text(text, "structural")


def _payload(text: str) -> Text:
    return Text(text, "payload")


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _tag(name: str, value: str, *, indent: str = "") -> Iterator[WireAtom]:
    # tag names are structural by the model's tag-name grammar
    # ([A-Za-z_][A-Za-z0-9_-]*): no glyph fragment can appear in them
    yield _structural(f"\n{indent}<{name}>")
    yield _payload(value)
    yield _structural(f"</{name}>")


def _system_atoms(system: SystemPrompt) -> Iterator[WireAtom]:
    """The compact system prompt block: one tag per line, absent tags omitted."""
    yield Control(ControlToken.SYS)
    if system.identity is not None:
        yield from _tag("identity", system.identity)
    if system.behavior is not None:
        yield from _tag("behavior", system.behavior)
    yield from _tag("effort", system.effort.value)
    if system.tools:
        yield _structural("\n<tools>")
        for spec in system.tools:
            yield _structural('\n  <tool name="')
            yield _payload(spec.name)
            yield _structural('">')
            dumped = spec.model_dump(mode="json")
            yield from _tag(
                "schema", _compact_json(dumped["json_schema"]), indent="    "
            )
            if spec.policy is not None:
                yield from _tag("policy", spec.policy, indent="    ")
            yield _structural("\n  </tool>")
        yield _structural("\n</tools>")
    for tag_name, value in system.extra_tags.items():
        yield from _tag(tag_name, value)
    if system.environment is not None:
        yield from _tag("environment", system.environment)
    yield _structural("\n")
    yield Control(ControlToken.SYS_END)


def _header_atoms(message: InputMessage | OutputMessage) -> Iterator[WireAtom]:
    """Space-separated ``key=value`` pairs: type first, typed keys, extras."""
    yield _structural(" type=")
    yield _payload(getattr(message, "type"))  # noqa: B009
    typed: list[tuple[str, str | None]] = []
    if isinstance(message, (ToolResult, ToolCall)):
        typed = [("id", message.id)]
    elif isinstance(message, Retrieval):
        typed = [("src", message.src)]
    elif isinstance(message, Attachment):
        typed = [("name", message.name), ("mime", message.mime)]
    for key, value in typed:
        if value is not None:
            yield _structural(f" {key}=")
            yield _payload(value)
    for key, value in message.extras.items():
        # extras keys are structural by the model's key grammar
        # ([A-Za-z0-9_]+): no glyph fragment can appear in them
        yield _structural(f" {key}=")
        yield _payload(value)


def _payload_atoms(message: InputMessage | OutputMessage) -> list[WireAtom]:
    """The payload as atoms; an empty list means an empty payload."""
    if isinstance(message, ToolCall):
        dumped = message.model_dump(mode="json")
        wire = {"name": dumped["name"], "args": dumped["args"]}
        return [_payload(_compact_json(wire))]
    payload = getattr(message, "payload")  # noqa: B009
    if isinstance(payload, str):
        return [_payload(payload)] if payload else []
    atoms: list[WireAtom] = []
    for part in payload:
        if isinstance(part, TextPart):
            if part.text:
                atoms.append(_payload(part.text))
        elif isinstance(part, ImagePart):
            atoms.append(Media("image"))
        elif isinstance(part, AudioPart):
            atoms.append(Media("audio"))
        else:
            assert_never(part)
    return atoms


def _message_atoms(message: InputMessage | OutputMessage) -> Iterator[WireAtom]:
    """``<|in|> HEADER <|hdr|> PAYLOAD <|/in|>`` with single-space joins."""
    is_input = isinstance(message, InputMessage)
    yield Control(ControlToken.IN if is_input else ControlToken.OUT)
    yield from _header_atoms(message)
    yield _structural(" ")
    yield Control(ControlToken.HDR)
    yield _structural(" ")
    payload = _payload_atoms(message)
    if payload:
        yield from payload
        yield _structural(" ")
    yield Control(ControlToken.IN_END if is_input else ControlToken.OUT_END)


def to_wire(conversation: Conversation) -> tuple[WireAtom, ...]:
    """Render *conversation* to its wire IR sequence.

    Items render exactly as given. Messages are separated by one blank
    line; a wait attaches to the preceding block on its own line. In the
    token projection, adjacent ``Text`` atoms must be merged before
    encoding so token boundaries match the string projection.
    """
    blocks: list[list[WireAtom]] = []
    if conversation.system is not None:
        blocks.append(list(_system_atoms(conversation.system)))
    for item in conversation.items:
        if isinstance(item, Wait):
            if blocks:
                blocks[-1].extend([_structural("\n"), Control(ControlToken.WAIT)])
            else:
                blocks.append([Control(ControlToken.WAIT)])
        else:
            blocks.append(list(_message_atoms(item)))
    atoms: list[WireAtom] = []
    for index, block in enumerate(blocks):
        if index:
            atoms.append(_structural("\n\n"))
        atoms.extend(block)
    return tuple(atoms)


def _atom_str(atom: WireAtom) -> str:
    if isinstance(atom, Control):
        return atom.token.value
    if isinstance(atom, Text):
        return atom.text
    token = ControlToken.IMAGE if atom.part == "image" else ControlToken.AUDIO
    return token.value


def render_str(conversation: Conversation) -> str:
    """Render *conversation* to the format's string form.

    Control tokens appear as their surface glyphs; payload text renders
    verbatim with no escaping or validation, so a payload that spells a
    control glyph is ambiguous in this projection (the planned token path will
    avoid this ambiguity). Rendering never reorders or normalizes; run
    ``conversation.normalized()`` first for canonical input order and
    ``check()`` for conformance.
    """
    return "".join(_atom_str(atom) for atom in to_wire(conversation))
