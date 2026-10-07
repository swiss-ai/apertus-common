"""One read-only view over the value and explicitly supplied earlier evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from apertus_common.checker._context import CheckContext
from apertus_common.checker._report import Location
from apertus_common.checker.rules._base import CheckValue
from apertus_common.model import (
    Conversation,
    Host,
    Message,
    SystemPrompt,
    ToolSpec,
    Wait,
)


@dataclass(frozen=True, slots=True)
class CheckView:
    context: CheckContext
    system: SystemPrompt | None
    system_location: Location
    systems: tuple[tuple[SystemPrompt, Location], ...]
    items: tuple[tuple[Message | Wait, Location], ...]
    tools: tuple[ToolSpec, ...] | None
    standalone: bool


def _same_system(left: SystemPrompt, right: SystemPrompt) -> bool:
    return all(
        getattr(left, name) == getattr(right, name)
        for name in SystemPrompt.model_fields
        if name != "loss_weight"
    )


def view(value: CheckValue, context: CheckContext | None) -> CheckView:
    standalone = not isinstance(value, Conversation)
    ctx = context or CheckContext(history="partial" if standalone else "complete")
    system = ctx.checkpoint.system if ctx.checkpoint else None
    system_location = Location("system", source="checkpoint")
    systems = [(system, system_location)] if system is not None else []
    items: list[tuple[Message | Wait, Location]] = []
    conversations: list[tuple[Conversation, Literal["value", "prefix"]]] = []
    if ctx.prefix is not None:
        conversations.append((ctx.prefix, "prefix"))
    if isinstance(value, Conversation):
        conversations.append((value, "value"))
    for conversation, source in conversations:
        location_source: Literal["value", "prefix"] = (
            "prefix" if source == "prefix" else "value"
        )
        if conversation.system is not None:
            if system is not None and not _same_system(system, conversation.system):
                raise ValueError("conflicting system configuration in checking context")
            system = conversation.system
            system_location = Location("system", source=location_source)
            systems.append((system, system_location))
        items.extend(
            (item, Location(item=index, source=location_source))
            for index, item in enumerate(conversation.items)
        )
    if isinstance(value, SystemPrompt):
        if system is not None and not _same_system(system, value):
            raise ValueError("conflicting system configuration in checking context")
        system = value
        system_location = Location("system")
        systems.append((system, system_location))
    elif isinstance(value, (Message, Wait)):
        items.append((value, Location(item=0)))
    if ctx.host_closures is not None:
        hosts = {
            (location.source, location.item)
            for item, location in items
            if isinstance(item, Host)
        }
        if any(
            (closure.source, closure.item) not in hosts for closure in ctx.host_closures
        ):
            raise ValueError("host closure evidence must identify a retained host item")
    tools = tuple(system.tools) if system is not None else None
    if isinstance(value, ToolSpec):
        tools = (value,)
    return CheckView(
        ctx, system, system_location, tuple(systems), tuple(items), tools, standalone
    )
