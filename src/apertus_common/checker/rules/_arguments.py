"""Schema compilation and arguments are deliberately opt-in heavy checks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

from apertus_common.checker._arguments import argument_findings, compile_schema
from apertus_common.checker._context import CheckContext
from apertus_common.checker._report import Unevaluated, Violation
from apertus_common.checker.rules._base import Checker, CheckValue, Rule
from apertus_common.checker.rules._view import view
from apertus_common.model import Call

if TYPE_CHECKING:
    from jsonschema.protocols import Validator


class ArgumentsChecker(Checker):
    rules: ClassVar[tuple[Rule, ...]] = (
        Rule("arguments/json", "call arguments are strict JSON objects"),
        Rule("arguments/schema", "arguments conform to the declared tool schema"),
        Rule(
            "arguments/broken-schema",
            "every declared schema is valid and locally resolvable",
        ),
        Rule("arguments/tool-name", "standalone argument checking uses the named tool"),
    )

    def check(
        self, value: CheckValue, context: CheckContext | None = None
    ) -> Iterable[Violation | Unevaluated]:
        current = view(value, context)
        validators: dict[str, Validator | None] = {}
        for index, tool in enumerate(current.tools or ()):
            try:
                validators[tool.name] = compile_schema(tool)
            except ValueError as exc:
                validators[tool.name] = None
                yield self.violation(
                    "arguments/broken-schema",
                    replace(current.system_location, field=f"tools[{index}].schema"),
                    str(exc),
                )
        for item, location in current.items:
            if isinstance(item, Call):
                yield from argument_findings(
                    item, validators.get(item.name), replace(location, field="payload")
                )
                if item.name not in validators:
                    yield Unevaluated(
                        "arguments/schema", location, "tool schema is unavailable"
                    )
