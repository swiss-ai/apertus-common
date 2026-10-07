"""Schema compilation and arguments are deliberately opt-in heavy checks."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

from apertus_common.checker._arguments import argument_findings, compile_schema
from apertus_common.checker._context import CheckContext
from apertus_common.checker._report import Unevaluated, Violation
from apertus_common.checker.rules._base import Checker, CheckValue, Rule
from apertus_common.checker.rules._view import view
from apertus_common.model import Call, ToolSpec

if TYPE_CHECKING:
    from jsonschema.protocols import Validator


class ArgumentsChecker(Checker):
    """Reusable checker with a bounded, process-local schema compilation cache."""

    def __init__(self, schema_cache_size: int = 256) -> None:
        if type(schema_cache_size) is not int or schema_cache_size < 0:
            raise ValueError("schema_cache_size must be a non-negative integer")
        self._schema_cache_size = schema_cache_size
        self._schemas: OrderedDict[str, Validator | str] = OrderedDict()

    def _validator(self, tool: ToolSpec) -> Validator:
        if not self._schema_cache_size:
            return compile_schema(tool)
        # Keep property order: it can affect the ordering of validation findings.
        key = json.dumps(
            tool.model_dump(mode="json", by_alias=True, include={"json_schema"})[
                "schema"
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        if key in self._schemas:
            cached = self._schemas[key]
            self._schemas.move_to_end(key)
        else:
            try:
                cached = compile_schema(tool)
            except ValueError as exc:
                cached = str(exc)
            self._schemas[key] = cached
            if len(self._schemas) > self._schema_cache_size:
                self._schemas.popitem(last=False)
        if isinstance(cached, str):
            raise ValueError(cached)  # noqa: TRY004
        return cached

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
                validators[tool.name] = self._validator(tool)
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
