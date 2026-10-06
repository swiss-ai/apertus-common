"""Observable history rules; missing history never becomes invented evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from apertus_common.checker._context import CallState, CheckContext, HostClosure
from apertus_common.checker._report import Location, Severity, Unevaluated, Violation
from apertus_common.checker.rules._base import Checker, CheckValue, Rule
from apertus_common.checker.rules._view import view
from apertus_common.model import (
    Call,
    Claim,
    Document,
    Host,
    InputMessage,
    OutputMessage,
    Reply,
    Result,
    Wait,
)


class ConversationChecker(Checker):
    rules: ClassVar[tuple[Rule, ...]] = (
        Rule("conversation/system", "complete histories have a system prompt"),
        Rule("conversation/declared-tool", "calls name a declared tool"),
        Rule(
            "conversation/counter", "calls use consecutive conversation-wide counters"
        ),
        Rule(
            "conversation/result",
            "each result closes exactly one matching pending call",
        ),
        Rule(
            "conversation/host-closure",
            "explicit host closures close matching pending calls",
        ),
        Rule("conversation/document-id", "document IDs are unique across the history"),
        Rule(
            "conversation/wait",
            "output or another wait after a wait requires intervening input",
        ),
        Rule(
            "conversation/claim-placement",
            "claims normally precede the replies they support",
            Severity.WARNING,
        ),
    )

    def check(
        self, value: CheckValue, context: CheckContext | None = None
    ) -> Iterable[Violation | Unevaluated]:
        current = view(value, context)
        ctx = current.context
        checkpoint = ctx.checkpoint
        complete = ctx.history == "complete"
        if not current.standalone and current.system is None:
            if complete:
                yield self.violation(
                    "conversation/system",
                    Location("system"),
                    "a complete history requires a system prompt",
                )
            else:
                yield Unevaluated(
                    "conversation/system",
                    Location("system"),
                    "system configuration before this fragment is unavailable",
                )
        next_counter = (
            checkpoint.next_call_counter if checkpoint else (0 if complete else None)
        )
        calls_known = complete or (
            checkpoint is not None and checkpoint.calls is not None
        )
        calls = (
            {call.counter: call for call in (checkpoint.calls or ())}
            if checkpoint
            else {}
        )
        document_ids: set[str] = (
            set(checkpoint.document_ids or ()) if checkpoint else set()
        )
        documents_known = complete or (
            checkpoint is not None and checkpoint.document_ids is not None
        )
        waiting = checkpoint.waiting if checkpoint else (False if complete else None)
        reply_seen = False
        closure_uncertain: set[int] = set()
        tools = None if current.tools is None else {tool.name for tool in current.tools}
        closures: dict[tuple[str, int | None], list[HostClosure]] = {}
        for closure in ctx.host_closures or ():
            closures.setdefault((closure.source, closure.item), []).append(closure)
        for item, location in current.items:
            if isinstance(item, Wait):
                if waiting is True:
                    yield self.violation(
                        "conversation/wait",
                        location,
                        "wait follows wait without new input",
                    )
                elif waiting is None:
                    yield Unevaluated(
                        "conversation/wait",
                        location,
                        "waiting state before this fragment is unavailable",
                    )
                waiting = True
                reply_seen = False
                continue
            if isinstance(item, InputMessage):
                waiting = False
            elif isinstance(item, OutputMessage):
                if waiting is True:
                    yield self.violation(
                        "conversation/wait",
                        location,
                        "output follows wait without new input",
                    )
                elif waiting is None:
                    yield Unevaluated(
                        "conversation/wait",
                        location,
                        "waiting state before this fragment is unavailable",
                    )
                waiting = False
            if isinstance(item, Reply):
                reply_seen = True
            elif isinstance(item, Claim) and reply_seen:
                yield self.violation(
                    "conversation/claim-placement",
                    location,
                    "claim appears after a reply in the same burst",
                )
            if isinstance(item, Document) and item.id is not None:
                if item.id in document_ids:
                    yield self.violation(
                        "conversation/document-id",
                        location,
                        f"duplicate document ID {item.id!r}",
                    )
                elif not documents_known:
                    yield Unevaluated(
                        "conversation/document-id",
                        location,
                        "earlier document IDs are unavailable",
                    )
                document_ids.add(item.id)
            if isinstance(item, Call):
                if tools is None:
                    yield Unevaluated(
                        "conversation/declared-tool",
                        location,
                        "tool declarations are unavailable",
                    )
                elif item.name not in tools:
                    yield self.violation(
                        "conversation/declared-tool",
                        location,
                        f"undeclared tool {item.name!r}",
                    )
                if item.counter in calls:
                    yield self.violation(
                        "conversation/counter",
                        location,
                        f"counter {item.counter} is already used",
                    )
                elif next_counter is None:
                    yield Unevaluated(
                        "conversation/counter",
                        location,
                        "next counter before this fragment is unavailable",
                    )
                elif item.counter != next_counter:
                    yield self.violation(
                        "conversation/counter",
                        location,
                        f"expected counter {next_counter}, received {item.counter}",
                    )
                if item.counter >= 0 and item.counter not in calls:
                    calls[item.counter] = CallState(item.counter, item.name)
                next_counter = max(item.counter + 1, next_counter or 0)
            elif isinstance(item, Result):
                call = calls.get(item.counter)
                if call is None:
                    if calls_known:
                        yield self.violation(
                            "conversation/result",
                            location,
                            "result has no preceding call",
                        )
                    else:
                        yield Unevaluated(
                            "conversation/result",
                            location,
                            "the matching call may precede this fragment",
                        )
                    if item.counter >= 0:
                        # The preceding call is unknown, but this closure is observed.
                        calls[item.counter] = CallState(
                            item.counter, item.name, "result"
                        )
                elif call.name != item.name:
                    yield self.violation(
                        "conversation/result",
                        location,
                        f"result name does not match {call.name!r}",
                    )
                elif call.status != "pending":
                    yield self.violation(
                        "conversation/result",
                        location,
                        f"call already closed by {call.status}",
                    )
                else:
                    if item.counter in closure_uncertain:
                        yield Unevaluated(
                            "conversation/result",
                            location,
                            "an earlier host item may already have closed this call",
                        )
                    calls[item.counter] = CallState(item.counter, item.name, "result")
            elif isinstance(item, Host):
                if ctx.host_closures is None:
                    pending = {
                        counter
                        for counter, call in calls.items()
                        if call.status == "pending"
                    }
                    if pending or not calls_known:
                        yield Unevaluated(
                            "conversation/host-closure",
                            location,
                            "host closure metadata is unavailable",
                        )
                        closure_uncertain.update(pending)
                for closure in closures.get((location.source, location.item), ()):
                    call = calls.get(closure.counter)
                    if call is None:
                        if calls_known:
                            yield self.violation(
                                "conversation/host-closure",
                                location,
                                "host closure has no preceding call",
                            )
                        else:
                            yield Unevaluated(
                                "conversation/host-closure",
                                location,
                                "the closed call may precede this fragment",
                            )
                        if closure.counter >= 0:
                            calls[closure.counter] = CallState(
                                closure.counter, closure.name, "host-closed"
                            )
                    elif call.name != closure.name or call.status != "pending":
                        yield self.violation(
                            "conversation/host-closure",
                            location,
                            "host closure must match a pending call's name and counter",
                        )
                    else:
                        calls[closure.counter] = CallState(
                            closure.counter, closure.name, "host-closed"
                        )
