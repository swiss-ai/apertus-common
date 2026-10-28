"""Profile-local rules, with no argument parsing or schema compilation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from itertools import chain
from typing import ClassVar

from apertus_common.checker._report import Unevaluated, Violation
from apertus_common.checker.rules._base import Rule
from apertus_common.checker.rules._view import CheckView, ViewChecker
from apertus_common.model import Call, Document, OutputMessage, Result, UnknownOutput


class ProfileChecker(ViewChecker):
    def __init__(self, *, allow_loss_weights: bool = False) -> None:
        self._allow_loss_weights = allow_loss_weights

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule("profile/known-output", "output types belong to the Apertus 2 profile"),
        Rule(
            "profile/nonnegative-counter", "call and result counters are non-negative"
        ),
        Rule(
            "profile/document-provenance",
            "documents declare supplying party and citation ID",
        ),
        Rule("profile/unique-tool-name", "declared tool names are unique"),
        Rule("profile/object-schema", "tool schemas declare object-root arguments"),
        Rule("profile/loss-weight", "training loss annotations require opt-in"),
        Rule("profile/closed-output", "output messages are closed"),
    )

    def check_view(self, current: CheckView) -> Iterable[Violation | Unevaluated]:
        if not self._allow_loss_weights:
            for weighted, location in chain(current.systems, current.items):
                if weighted.loss_weight is not None:
                    yield self.violation(
                        "profile/loss-weight",
                        replace(location, field="loss_weight"),
                        "loss weights require allow_loss_weights=True",
                    )
        for item, location in current.items:
            if isinstance(item, OutputMessage) and item.open:
                yield self.violation(
                    "profile/closed-output", location, "output message is open"
                )
            if isinstance(item, UnknownOutput):
                yield self.violation(
                    "profile/known-output",
                    location,
                    f"unsupported output type {item.output_type!r}",
                )
            if isinstance(item, (Call, Result)) and item.counter < 0:
                yield self.violation(
                    "profile/nonnegative-counter",
                    replace(location, field="counter"),
                    "counter must be non-negative",
                )
            if isinstance(item, Document) and (item.from_ is None or item.id is None):
                yield self.violation(
                    "profile/document-provenance",
                    location,
                    "documents require both from and id",
                )
        names: set[str] = set()
        for index, tool in enumerate(current.tools or ()):
            location = replace(current.system_location, field=f"tools[{index}]")
            if tool.name in names:
                yield self.violation(
                    "profile/unique-tool-name",
                    location,
                    f"duplicate tool name {tool.name!r}",
                )
            names.add(tool.name)
            if tool.json_schema.get("type") != "object":
                yield self.violation(
                    "profile/object-schema",
                    location,
                    "tool argument schema must declare type='object'",
                )
