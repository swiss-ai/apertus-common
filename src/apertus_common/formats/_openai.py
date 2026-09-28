"""OpenAI chat-completions messages to and from the typed model.

A deliberately lossy adapter: the OpenAI format has one fixed role set and
no room for most of the Apertus cast. ``from_openai`` lifts any valid
message list (round-trip caveats on the function); ``to_openai`` maps back
the subset stated on it and raises ``AdapterError`` for the rest. The
planned response side (``finish_reason``, streaming chunks) belongs in a
serving adapter over the streaming parser and is not part of this module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast, overload

from pydantic import ValidationError

from apertus_common.checker import ConformanceReport
from apertus_common.errors import AdapterError
from apertus_common.formats._json import finish
from apertus_common.model import (
    Assistant,
    Conversation,
    Effort,
    Harness,
    InputMessage,
    OutputMessage,
    SystemPrompt,
    TextPart,
    Think,
    ToolCall,
    ToolResult,
    ToolSpec,
    User,
    Wait,
)

_Message = Mapping[str, Any]
_SYSTEM_ROLES = frozenset({"system", "developer"})


def _fail(index: int, reason: str) -> AdapterError:
    return AdapterError("message", index, reason)


def _mapping(value: object, index: int, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail(index, f"{what} is not an object")
    return cast(Mapping[str, object], value)


def _text(value: object, index: int, key: str) -> str:
    """*value* as text: a string, ``None`` as empty, or joined text parts."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces: list[str] = []
        for raw in cast(list[object], value):
            part = _mapping(raw, index, f"{key} part")
            if part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise _fail(index, f"{key} may only contain text parts here")
            pieces.append(cast(str, part["text"]))
        return "".join(pieces)
    raise _fail(index, f"{key} must be a string or a list of parts")


def _user_payload(value: object, index: int) -> str | list[object]:
    """User content: a string or text parts; unsupported parts are rejected."""
    if not isinstance(value, list):
        return _text(value, index, "content")
    parts: list[object] = []
    for raw in cast(list[object], value):
        part = _mapping(raw, index, "content part")
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            parts.append(_construct(index, TextPart, text=part["text"]))
        else:
            raise _fail(index, f"unsupported content part type {kind!r}")
    return parts


def _construct[T](index: int, cls: type[T], **fields: object) -> T:
    try:
        return cls(**fields)
    except ValidationError as error:
        raise _fail(index, str(error)) from error


def _tool_calls(value: object, index: int) -> list[ToolCall]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise _fail(index, "tool_calls must be a list")
    calls: list[ToolCall] = []
    for raw in cast(list[object], value):
        call = _mapping(raw, index, "tool_calls entry")
        call_id = call.get("id")
        function = call.get("function")
        if not isinstance(call_id, str) or not isinstance(function, Mapping):
            raise _fail(index, "tool_calls entry needs id and function")
        function_map = cast(Mapping[str, object], function)
        name = function_map.get("name")
        if not isinstance(name, str):
            raise _fail(index, "tool_calls entry needs function.name")
        arguments = function_map.get("arguments")
        args: object = arguments
        if arguments is None or arguments == "":
            args = {}
        elif isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise _fail(
                    index, f"tool call {call_id!r} arguments are not JSON"
                ) from error
        calls.append(_construct(index, ToolCall, id=call_id, name=name, args=args))
    return calls


def _tool_specs(tools: Sequence[_Message] | None) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for position, raw in enumerate(tools or ()):
        label = f"tools[{position}]"
        try:
            tool = _mapping(raw, -1, label)
            function = tool.get("function")
            if tool.get("type", "function") != "function" or not isinstance(
                function, Mapping
            ):
                raise _fail(-1, f"{label} is not a function tool")
            function_map = cast(Mapping[str, object], function)
            name = function_map.get("name")
            if not isinstance(name, str):
                raise _fail(-1, f"{label} needs function.name")
            parameters = function_map.get("parameters")
            description = function_map.get("description")
            specs.append(
                ToolSpec(
                    name=name,
                    json_schema=cast(Any, parameters)
                    if parameters is not None
                    else {"type": "object"},
                    policy=description if isinstance(description, str) else None,
                )
            )
        except AdapterError as error:
            raise AdapterError("system", -1, error.reason) from None
        except ValidationError as error:
            raise AdapterError("system", -1, f"{label}: {error}") from error
    return specs


@overload
def from_openai(
    messages: Sequence[_Message],
    tools: Sequence[_Message] | None = None,
    *,
    effort: Effort | str = Effort.MEDIUM,
    continue_final_message: bool = False,
    normalize: bool = True,
    check: Literal[False] = False,
    strict: bool = False,
) -> Conversation: ...


@overload
def from_openai(
    messages: Sequence[_Message],
    tools: Sequence[_Message] | None = None,
    *,
    effort: Effort | str = Effort.MEDIUM,
    continue_final_message: bool = False,
    normalize: bool = True,
    check: Literal[True],
    strict: bool = False,
) -> tuple[Conversation, ConformanceReport]: ...


@overload
def from_openai(
    messages: Sequence[_Message],
    tools: Sequence[_Message] | None = None,
    *,
    effort: Effort | str = Effort.MEDIUM,
    continue_final_message: bool = False,
    normalize: bool = True,
    check: bool,
    strict: bool = False,
) -> Conversation | tuple[Conversation, ConformanceReport]: ...


def from_openai(
    messages: Sequence[_Message],
    tools: Sequence[_Message] | None = None,
    *,
    effort: Effort | str = Effort.MEDIUM,
    continue_final_message: bool = False,
    normalize: bool = True,
    check: bool = False,
    strict: bool = False,
) -> Conversation | tuple[Conversation, ConformanceReport]:
    """Build a ``Conversation`` from OpenAI chat-completions *messages*.

    Mapping: a leading ``system``/``developer`` message becomes the system
    prompt's ``behavior`` (there is always a system prompt, since ``effort``
    is required; pass the request's ``reasoning_effort`` as *effort*); a later
    ``system``/``developer`` becomes ``Harness``; ``user`` content maps to
    text parts; ``assistant`` yields, in order,
    ``Think`` from ``reasoning_content``, ``Assistant`` from ``content``, and
    one ``ToolCall`` per ``tool_calls`` entry (JSON-string ``arguments``
    parsed, ids kept byte for byte); ``tool`` becomes ``ToolResult``. *tools*
    map to ``ToolSpec`` (``parameters`` as schema, ``description`` as policy).
    ``name`` fields are ignored.

    A ``Wait`` closes each run of assistant messages, including a trailing
    one unless ``continue_final_message=True`` (prefill: the last burst stays
    open). Ingest flags ``normalize``, ``check``, ``strict`` as documented on
    the ``formats`` package. Round trips normalize representations:
    singleton user text-part lists become strings, adjacent assistant messages
    may merge, empty reasoning/content segments may be omitted, and
    ``developer`` comes back as ``system``. Assistant
    ``content`` is always present (``null`` when absent), ``tool_calls``
    carry ``type`` and compact re-serialized ``arguments``,
    ``reasoning_effort`` is always emitted, ``name``/``strict`` and unknown keys
    are dropped, and the default normalization
    may move a later ``system`` notice ahead of the ``user`` or ``tool``
    messages it followed (``normalize=False`` keeps arrival order). A
    message that cannot be mapped raises ``AdapterError`` (``where="message"``,
    its index); an invalid *effort* or *tools* entry raises with
    ``where="system"``.
    """
    specs = _tool_specs(tools)
    behavior: str | None = None
    items: list[InputMessage | OutputMessage | Wait] = []
    open_run = False
    for index, raw in enumerate(messages):
        message = _mapping(raw, index, "message")
        role = message.get("role")
        if role == "assistant":
            open_run = True
            think = _text(message.get("reasoning_content"), index, "reasoning_content")
            content = _text(message.get("content"), index, "content")
            calls = _tool_calls(message.get("tool_calls"), index)
            if think:
                items.append(_construct(index, Think, payload=think))
            if content or (not think and not calls):
                items.append(_construct(index, Assistant, payload=content))
            items.extend(calls)
            continue
        if open_run:
            items.append(Wait())
            open_run = False
        if role in _SYSTEM_ROLES:
            text = _text(message.get("content"), index, "content")
            if index == 0:
                behavior = text or None
            else:
                items.append(_construct(index, Harness, payload=text))
        elif role == "user":
            payload = _user_payload(message.get("content"), index)
            items.append(_construct(index, User, payload=payload))
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str):
                raise _fail(index, "tool message needs tool_call_id")
            payload = _text(message.get("content"), index, "content")
            items.append(_construct(index, ToolResult, id=call_id, payload=payload))
        else:
            raise _fail(index, f"unsupported role {role!r}")
    if open_run and not continue_final_message:
        items.append(Wait())
    try:
        system = SystemPrompt(behavior=behavior, effort=Effort(effort), tools=specs)
    except ValueError as error:  # includes pydantic ValidationError
        raise AdapterError("system", -1, str(error)) from error
    conversation = Conversation.model_validate({"system": system, "items": items})
    return finish(conversation, normalize=normalize, check=check, strict=strict)


def _system_message(system: SystemPrompt) -> dict[str, Any] | None:
    if system.extra_tags:
        raise AdapterError("system", -1, "extra_tags have no OpenAI counterpart")
    text = "\n\n".join(
        part for part in (system.identity, system.behavior, system.environment) if part
    )
    return {"role": "system", "content": text} if text else None


def _tool_dicts(system: SystemPrompt) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for spec in system.tools:
        function: dict[str, Any] = {"name": spec.name}
        if spec.policy is not None:
            function["description"] = spec.policy
        function["parameters"] = spec.model_dump(mode="json")["json_schema"]
        tools.append({"type": "function", "function": function})
    return tools


def _user_content(user: User) -> str | list[dict[str, Any]]:
    parts = list(user.payload)
    if len(parts) == 1:
        return parts[0].text
    return [{"type": "text", "text": part.text} for part in parts]


class _Burst:
    """Groups a run of output messages into OpenAI assistant dicts.

    One dict holds at most one reasoning, one content, and any number of
    calls, in that order; an item that would repeat a slot or place
    reasoning after content starts a new dict.
    """

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink
        self._open = False
        self._reasoning: str | None = None
        self._content: str | None = None
        self._calls: list[dict[str, Any]] = []

    def flush(self) -> None:
        if not self._open:
            return
        message: dict[str, Any] = {"role": "assistant"}
        if self._reasoning is not None:
            message["reasoning_content"] = self._reasoning
        message["content"] = self._content
        if self._calls:
            message["tool_calls"] = self._calls
        self._sink.append(message)
        self._open = False
        self._reasoning = self._content = None
        self._calls = []

    def think(self, text: str) -> None:
        if self._reasoning is not None or self._content is not None or self._calls:
            self.flush()
        self._open = True
        self._reasoning = text

    def content(self, text: str) -> None:
        if self._content is not None or self._calls:
            self.flush()
        self._open = True
        self._content = text

    def call(self, call: ToolCall) -> None:
        self._open = True
        args = call.model_dump(mode="json")["args"]
        self._calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        args, separators=(",", ":"), ensure_ascii=False
                    ),
                },
            }
        )


def to_openai(conversation: Conversation) -> dict[str, Any]:
    """Map *conversation* to an OpenAI chat-completions request dict.

    Returns ``{"messages": [...]}`` plus ``"tools"`` when the system prompt
    declares tools and ``"reasoning_effort"`` when there is a system prompt.
    Supported subset: ``Harness`` (as ``system``), ``User``, ``ToolResult``
    (as ``tool``), ``Think`` (``reasoning_content``), ``Assistant``
    (``content``), ``ToolCall`` (``tool_calls``, compact JSON ``arguments``,
    ids byte for byte), and ``Wait`` (closes the assistant run). The system
    prompt collapses identity, behavior, and environment into one ``system``
    string (lossy on tag boundaries) and its tools into ``tools``. A burst is
    one assistant message while it reads reasoning, then content, then calls;
    each further switch (content before reasoning, a second content) starts
    another assistant message, so reasoning never enters ``content``.

    Everything else has no counterpart and raises ``AdapterError`` rather
    than being dropped or relabelled: ``Retrieval``, ``Event``,
    ``Attachment``, ``Document``, ``GenericInput``, ``GenericOutput``,
    ``VerifiableAnswer``, non-empty header ``extras``, and system
    ``extra_tags``. Filter or convert such items first.
    """
    messages: list[dict[str, Any]] = []
    result: dict[str, Any] = {"messages": messages}
    if conversation.system is not None:
        system_message = _system_message(conversation.system)
        if system_message is not None:
            messages.append(system_message)
        tools = _tool_dicts(conversation.system)
        if tools:
            result["tools"] = tools
        result["reasoning_effort"] = conversation.system.effort.value
    burst = _Burst(messages)
    for index, item in enumerate(conversation.items):
        if isinstance(item, Wait):
            burst.flush()
            continue
        if item.extras:
            raise AdapterError(
                "item", index, "header extras have no OpenAI counterpart"
            )
        if isinstance(item, Think):
            burst.think(item.payload)
        elif isinstance(item, Assistant):
            burst.content(item.payload)
        elif isinstance(item, ToolCall):
            burst.call(item)
        else:
            burst.flush()
            if isinstance(item, Harness):
                messages.append({"role": "system", "content": item.payload})
            elif isinstance(item, User):
                messages.append({"role": "user", "content": _user_content(item)})
            elif isinstance(item, ToolResult):
                messages.append(
                    {"role": "tool", "tool_call_id": item.id, "content": item.payload}
                )
            else:
                raise AdapterError(
                    "item", index, f"{item.type!r} messages have no OpenAI counterpart"
                )
    burst.flush()
    return result
