"""Explicit OpenAI text/tool mapping, independent of conformance checking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel, ValidationError

from apertus_common.errors import AdapterError
from apertus_common.formats._stream import OpenAIStreamMapper
from apertus_common.formats._types import (
    Associations,
    MappingDiagnostic,
    MappingPolicy,
    MappingResult,
    OpenAIAssistantMessage,
    OpenAIMappingContext,
    OpenAIMessage,
    OpenAIRequest,
    OpenAITool,
)
from apertus_common.model import (
    Call,
    Claim,
    Conversation,
    JsonObject,
    OutputMessage,
    Reply,
    Result,
    SystemPrompt,
    Think,
    Thinking,
    ToolSpec,
    User,
    Wait,
)
from apertus_common.model._conversation import ConversationItem
from apertus_common.parsing import (
    MessageCompleted,
    MessageStart,
    PayloadDelta,
    WaitReached,
)


def _fail(index: int, message: str) -> AdapterError:
    return AdapterError("system" if index < 0 else "message", index, message)


def _object(value: object, index: int) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in cast(Mapping[object, object], value)
    ):
        raise _fail(index, "expected an object with string keys")
    return cast(Mapping[str, object], value)


def _keys(value: Mapping[str, object], allowed: set[str], index: int) -> None:
    extra = value.keys() - allowed
    if extra:
        raise _fail(index, f"unsupported fields: {sorted(extra)}")


def _string(value: object, index: int, label: str) -> str:
    if not isinstance(value, str):
        raise _fail(index, f"{label} must be a string")
    return value


def _construct[M: BaseModel](cls: type[M], index: int, **values: object) -> M:
    try:
        return cls.model_validate(values)
    except ValidationError as error:
        raise _fail(index, str(error)) from error


def _text(
    value: object,
    index: int,
    diagnostics: list[MappingDiagnostic],
    *,
    nullable: bool = False,
) -> str | None:
    if value is None and nullable:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for raw in cast(list[object], value):
            part = _object(raw, index)
            _keys(part, {"type", "text"}, index)
            if part.get("type") != "text":
                raise _fail(index, "only text content parts are supported")
            parts.append(_string(part.get("text"), index, "text part"))
        diagnostics.append(
            MappingDiagnostic(
                "text-parts-joined", index, "Text parts normalized to a string"
            )
        )
        return "".join(parts)
    raise _fail(index, "content must be text")


def _tools(tools: Sequence[Mapping[str, object]]) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for raw in tools:
        tool = _object(raw, -1)
        _keys(tool, {"type", "function"}, -1)
        if tool.get("type") != "function":
            raise _fail(-1, "only function tools are supported")
        function = _object(tool.get("function"), -1)
        _keys(function, {"name", "description", "parameters"}, -1)
        specs.append(
            _construct(
                ToolSpec,
                -1,
                name=_string(function.get("name"), -1, "tool name"),
                description=_string(
                    function.get("description", ""), -1, "tool description"
                ),
                schema=function.get("parameters", {"type": "object", "properties": {}}),
            )
        )
    return specs


def from_openai(
    messages: Sequence[Mapping[str, object]],
    *,
    tools: Sequence[Mapping[str, object]] | None = None,
    reasoning_effort: str | Thinking | None = None,
    system_defaults: Mapping[str, object] | SystemPrompt | None = None,
    context: OpenAIMappingContext | None = None,
    policy: MappingPolicy | None = None,
) -> MappingResult[Conversation]:
    """Map the supported request subset without repairing or checking content."""
    policy = policy or MappingPolicy()
    ids = Associations(context)
    diagnostics: list[MappingDiagnostic] = []
    items: list[ConversationItem] = []
    leading: list[str] = []
    leading_roles: list[str] = []
    assistant_run = False
    seen_calls: set[str] = set()
    for index, raw in enumerate(messages):
        message = _object(raw, index)
        role = message.get("role")
        if role in ("system", "developer"):
            _keys(message, {"role", "content"}, index)
            if items or assistant_run:
                raise _fail(index, "system/developer instructions must be leading")
            if (role == "developer" or leading) and not policy.merge_system_messages:
                raise _fail(
                    index,
                    "developer/multiple system messages require merge_system_messages",
                )
            text = _text(message.get("content"), index, diagnostics)
            assert text is not None
            leading.append(text)
            leading_roles.append(role)
            continue
        if assistant_run and role != "assistant":
            items.append(Wait())
            assistant_run = False
        if role == "user":
            _keys(message, {"role", "content"}, index)
            items.append(
                _construct(
                    User,
                    index,
                    payload=_text(message.get("content"), index, diagnostics),
                )
            )
        elif role == "tool":
            _keys(message, {"role", "content", "tool_call_id"}, index)
            call_id = _string(message.get("tool_call_id"), index, "tool_call_id")
            association = ids.by_id.get(call_id)
            if association is None:
                raise _fail(index, "tool result has no known call association")
            items.append(
                _construct(
                    Result,
                    index,
                    name=association.name,
                    counter=association.counter,
                    payload=_text(message.get("content"), index, diagnostics),
                )
            )
        elif role == "assistant":
            _keys(
                message, {"role", "content", "reasoning_content", "tool_calls"}, index
            )
            assistant_run = True
            initial = len(items)
            reasoning = _text(
                message.get("reasoning_content"), index, diagnostics, nullable=True
            )
            content = _text(message.get("content"), index, diagnostics, nullable=True)
            if reasoning is not None:
                items.append(_construct(Think, index, payload=reasoning))
            if content is not None:
                items.append(_construct(Reply, index, payload=content))
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list):
                raise _fail(index, "tool_calls must be an array")
            for raw_call in cast(list[object], calls):
                call = _object(raw_call, index)
                _keys(call, {"id", "type", "function"}, index)
                if call.get("type") != "function":
                    raise _fail(index, "only function calls are supported")
                call_id = _string(call.get("id"), index, "call id")
                if not call_id or call_id in seen_calls:
                    raise _fail(
                        index, "call IDs must be nonempty and unique in the input"
                    )
                seen_calls.add(call_id)
                function = _object(call.get("function"), index)
                _keys(function, {"name", "arguments"}, index)
                name = _string(function.get("name"), index, "function.name")
                arguments = _string(
                    function.get("arguments"), index, "function.arguments"
                )
                association = ids.external(call_id, name, index)
                items.append(
                    _construct(
                        Call,
                        index,
                        name=name,
                        counter=association.counter,
                        payload=arguments,
                    )
                )
            if initial == len(items):
                items.append(Reply(payload=""))
        else:
            raise _fail(index, f"unsupported role {role!r}")
    if assistant_run:
        items.append(Wait())
    if len(leading) > 1 or "developer" in leading_roles:
        diagnostics.append(
            MappingDiagnostic(
                "system-merged",
                0,
                "System/developer role distinctions merged into behavior",
                True,
            )
        )
    try:
        settings: dict[str, object] = (
            dict(system_defaults.model_dump())
            if isinstance(system_defaults, SystemPrompt)
            else dict(system_defaults or {})
        )
        _keys(
            settings, {"thinking", "behavior", "identity", "environment", "tools"}, -1
        )
        supplied: dict[str, object] = {}
        if leading:
            supplied["behavior"] = "\n\n".join(leading)
        if reasoning_effort is not None:
            supplied["thinking"] = reasoning_effort
        if tools is not None:
            supplied["tools"] = _tools(tools)
        # Validate defaults and supplied values before comparing typed meaning.
        default_system = SystemPrompt.model_validate(
            {"thinking": Thinking.MEDIUM, **settings}
        )
        candidate = SystemPrompt.model_validate(
            {**default_system.model_dump(), **supplied}
        )
        for key in settings.keys() & supplied.keys():
            if getattr(default_system, key) != getattr(candidate, key):
                raise AdapterError("system", -1, f"conflicting {key} settings")  # noqa: TRY301
        conversation = Conversation(system=candidate, items=items)
    except AdapterError:
        raise
    except (ValidationError, TypeError, ValueError) as error:
        raise AdapterError("system", -1, str(error)) from error
    return MappingResult(conversation, tuple(diagnostics), ids.snapshot())


def to_openai_output(
    output: Conversation,
    *,
    context: OpenAIMappingContext | None = None,
    policy: MappingPolicy | None = None,
) -> MappingResult[OpenAIAssistantMessage]:
    """Project an output-only fragment using the same mapper as streaming."""
    if output.system is not None:
        raise AdapterError(
            "system", -1, "output fragment must not contain a system prompt"
        )
    mapper = OpenAIStreamMapper(context, policy)
    ordinal = 0
    for index, item in enumerate(output.items):
        if isinstance(item, Wait):
            mapper.push(WaitReached())
            continue
        if not isinstance(item, OutputMessage):
            raise AdapterError(
                "item", index, "output fragment contains an input message"
            )
        headers: JsonObject = (
            {"name": item.name, "counter": item.counter}
            if isinstance(item, Call)
            else {}
        )
        mapper.push(MessageStart(ordinal, item.type, headers))
        mapper.push(PayloadDelta(ordinal, item.payload))
        mapper.push(MessageCompleted(ordinal, item))
        ordinal += 1
    return mapper.finish()


def to_openai(
    conversation: Conversation,
    *,
    context: OpenAIMappingContext | None = None,
    policy: MappingPolicy | None = None,
) -> MappingResult[OpenAIRequest]:
    """Map supported history, rejecting unrepresentable data unless permitted."""
    policy = policy or MappingPolicy()
    ids = Associations(context)
    messages: list[OpenAIMessage] = []
    diagnostics: list[MappingDiagnostic] = []
    request: OpenAIRequest = {"messages": messages}
    if conversation.system is not None:
        system = conversation.system
        if (
            system.identity is not None
            or system.environment is not None
            or any(t.policy is not None for t in system.tools)
        ):
            raise AdapterError(
                "system",
                -1,
                "identity/environment/tool policy have no lossless OpenAI counterpart",
            )
        if system.behavior is not None:
            messages.append({"role": "system", "content": system.behavior})
        request["reasoning_effort"] = system.thinking.value
        tools: list[OpenAITool] = []
        for tool in system.tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": cast(
                            dict[str, object],
                            tool.model_dump(mode="json", by_alias=True)["schema"],
                        ),
                    },
                }
            )
        request["tools"] = tools
    run: list[ConversationItem] = []
    last_kind = -1
    after_wait = False
    seen_counters: set[int] = set()

    def flush() -> None:
        nonlocal ids, last_kind
        if not run:
            return
        result = to_openai_output(
            Conversation(items=list(run)), context=ids.snapshot(), policy=policy
        )
        messages.append(cast(OpenAIMessage, result.value))
        diagnostics.extend(result.diagnostics)
        ids = Associations(result.context)
        run.clear()
        last_kind = -1

    for index, item in enumerate(conversation.items):
        if isinstance(item, Wait):
            if not run:
                if not policy.merge_assistant_messages:
                    raise AdapterError(
                        "item", index, "bare/consecutive wait cannot be represented"
                    )
                diagnostics.append(
                    MappingDiagnostic("wait-omitted", index, "Bare wait omitted", True)
                )
            else:
                run.append(item)
                flush()
            after_wait = True
        elif isinstance(item, (Think, Reply, Call, Claim)):
            if after_wait:
                if not policy.merge_assistant_messages:
                    raise AdapterError(
                        "item",
                        index,
                        "consecutive assistant bursts require merging policy",
                    )
                diagnostics.append(
                    MappingDiagnostic(
                        "burst-boundary-lost",
                        index,
                        "Adjacent assistant burst boundary omitted",
                        True,
                    )
                )
                after_wait = False
            if isinstance(item, Call):
                if item.counter in seen_counters:
                    raise AdapterError(
                        "item", index, "duplicate call counter in history"
                    )
                seen_counters.add(item.counter)
            kind = 0 if isinstance(item, Think) else 1 if isinstance(item, Reply) else 2
            if not isinstance(item, Claim) and (
                kind < last_kind or (kind == last_kind and kind < 2)
            ):
                flush()
            run.append(item)
            if not isinstance(item, Claim):
                last_kind = kind
        else:
            if run:
                if not policy.merge_assistant_messages:
                    raise AdapterError(
                        "item",
                        index,
                        "interleaved input before wait requires merging policy",
                    )
                diagnostics.append(
                    MappingDiagnostic(
                        "burst-boundary-inserted",
                        index,
                        "Chat mapping inserts an assistant turn boundary",
                        True,
                    )
                )
                flush()
            after_wait = False
            if isinstance(item, User):
                messages.append({"role": "user", "content": item.payload})
            elif isinstance(item, Result):
                association = ids.native(item.counter, item.name, index, create=False)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": association.id,
                        "content": item.payload,
                    }
                )
            else:
                raise AdapterError(
                    "item", index, f"unsupported message type {item.type!r}"
                )
    if run:
        if not policy.merge_assistant_messages:
            raise AdapterError(
                "item",
                len(conversation.items) - 1,
                "unfinished assistant burst requires merging policy",
            )
        diagnostics.append(
            MappingDiagnostic(
                "burst-boundary-inserted",
                len(conversation.items) - 1,
                "Chat history closes the partial assistant burst",
                True,
            )
        )
        flush()
    return MappingResult(request, tuple(diagnostics), ids.snapshot())
