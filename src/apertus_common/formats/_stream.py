"""Project native parser events to one OpenAI assistant response."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from pydantic import ValidationError

from apertus_common.errors import AdapterError
from apertus_common.formats._types import (
    Associations,
    MappingDiagnostic,
    MappingPolicy,
    MappingResult,
    OpenAIAssistantMessage,
    OpenAIDelta,
    OpenAIMappingContext,
    OpenAIToolCall,
    loss_weight_diagnostic,
)
from apertus_common.model import Call
from apertus_common.parsing import (
    MessageCompleted,
    MessageStart,
    ParseEvent,
    PayloadDelta,
    WaitReached,
)


class OpenAIStreamMapper:
    """One event stream, explicit loss policy, no parsing or tool execution."""

    def __init__(
        self,
        context: OpenAIMappingContext | None = None,
        policy: MappingPolicy | None = None,
    ) -> None:
        self._ids = Associations(context)
        self._policy = policy or MappingPolicy()
        self._diagnostics: list[MappingDiagnostic] = []
        self._diagnostic_cursor = 0
        self._active: MessageStart | None = None
        self._parts: list[str] = []
        self._reasoning: list[str] = []
        self._content: list[str] = []
        self._calls: list[OpenAIToolCall] = []
        self._seen: list[str] = []
        self._counter_seen: set[int] = set()
        self._index = 0
        self._waiting = False
        self._failed = False
        self._result: MappingResult[OpenAIAssistantMessage] | None = None

    def _fail(self, detail: str) -> NoReturn:
        self._failed = True
        raise AdapterError("item", self._index, detail)

    def push(self, event: ParseEvent) -> Sequence[OpenAIDelta]:
        if self._failed or self._result is not None or self._waiting:
            self._fail("event after mapper termination")
        if isinstance(event, WaitReached):
            if self._active is not None:
                self._fail("wait inside an unfinished message")
            self._waiting = True
            return []
        if isinstance(event, MessageStart):
            if self._active is not None or event.index != self._index:
                self._fail("message start is out of sequence")
            kind = event.type
            expected_headers: set[str] = (
                {"name", "counter"} if kind == "call" else set()
            )
            if set(event.headers) != expected_headers:
                self._fail("unexpected or missing event header fields")
            if kind == "claim" and self._policy.omit_claims:
                self._diagnostics.append(
                    MappingDiagnostic(
                        "claim-omitted",
                        event.index,
                        "Claim omitted from chat output",
                        True,
                    )
                )
            elif kind not in {"think", "reply", "call"}:
                self._fail(f"unsupported output type {kind!r}")
            else:
                order = {"think": 0, "reply": 1, "call": 2}
                previous = [x for x in self._seen if x != "claim"]
                lossy = (kind != "call" and kind in previous) or bool(
                    previous and order[kind] < order[previous[-1]]
                )
                if lossy:
                    if not self._policy.merge_assistant_messages:
                        self._fail(
                            "multiple or reordered segments require merge_assistant_messages"
                        )
                    self._diagnostics.append(
                        MappingDiagnostic(
                            "assistant-merged",
                            event.index,
                            "Native segment boundaries/order merged",
                            True,
                        )
                    )
            self._active = event
            self._seen.append(kind)
            self._parts = []
            if kind == "call":
                name, counter = event.headers.get("name"), event.headers.get("counter")
                if not isinstance(name, str) or type(counter) is not int:
                    self._fail("invalid call header in event")
                assert isinstance(name, str) and isinstance(counter, int)
                if counter in self._counter_seen:
                    self._fail("duplicate call counter in output")
                self._counter_seen.add(counter)
                try:
                    association = self._ids.native(counter, name, event.index)
                except AdapterError:
                    self._failed = True
                    raise
                except ValidationError as error:
                    self._failed = True
                    raise AdapterError(
                        "item", event.index, "invalid tool association"
                    ) from error
                self._calls.append(
                    {
                        "id": association.id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                )
                return [
                    {
                        "tool_calls": [
                            {
                                "index": len(self._calls) - 1,
                                "id": association.id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    }
                ]
            return []
        if self._active is None or event.index != self._active.index:
            self._fail("event has no matching message start")
        assert self._active is not None
        kind = self._active.type
        if isinstance(event, PayloadDelta):
            self._parts.append(event.text)
            if kind == "think":
                return [{"reasoning_content": event.text}]
            if kind == "reply":
                return [{"content": event.text}]
            if kind == "call":
                return [
                    {
                        "tool_calls": [
                            {
                                "index": len(self._calls) - 1,
                                "function": {"arguments": event.text},
                            }
                        ]
                    }
                ]
            return []
        assert isinstance(event, MessageCompleted)
        if event.message.open:
            self._fail("open messages have no OpenAI mapping")
        try:
            diagnostic = loss_weight_diagnostic(
                event.message.loss_weight, self._policy, event.index
            )
        except AdapterError:
            self._failed = True
            raise
        if diagnostic is not None:
            self._diagnostics.append(diagnostic)
        text = "".join(self._parts)
        if event.message.type != kind or event.message.payload != text:
            self._fail("completed message differs from its provisional events")
        if kind == "call":
            if (
                not isinstance(event.message, Call)
                or event.message.name != self._active.headers.get("name")
                or event.message.counter != self._active.headers.get("counter")
            ):
                self._fail("completed call differs from its header")
            self._calls[-1]["function"]["arguments"] = text
        elif kind == "think":
            self._reasoning.append(text)
        elif kind == "reply":
            self._content.append(text)
        self._active = None
        self._parts = []
        self._index += 1
        return []

    def take_diagnostics(self) -> list[MappingDiagnostic]:
        result = self._diagnostics[self._diagnostic_cursor :]
        self._diagnostic_cursor = len(self._diagnostics)
        return result

    def finish(self) -> MappingResult[OpenAIAssistantMessage]:
        if self._failed or self._active is not None:
            self._fail("cannot finish failed mapper or unfinished message")
        if self._result is None:
            result: OpenAIAssistantMessage = {
                "role": "assistant",
                "content": "".join(self._content) if self._content else None,
            }
            if self._reasoning:
                result["reasoning_content"] = "".join(self._reasoning)
            if self._calls:
                result["tool_calls"] = self._calls
            self._result = MappingResult(
                result, tuple(self._diagnostics), self._ids.snapshot()
            )
        return self._result
