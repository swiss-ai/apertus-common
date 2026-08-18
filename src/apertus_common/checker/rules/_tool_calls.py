"""The tool call payload against the declared inventory.

The format spec requires that every tool call names a declared tool and
that its args conform to that tool's declared JSON Schema.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar, NoReturn, cast

from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import NoSuchResource

from apertus_common.checker._report import Location, Severity, Violation
from apertus_common.checker.rules._base import Checker, Rule
from apertus_common.model import Conversation, ToolCall


def _refuse_retrieval(uri: str) -> NoReturn:
    # conformance checking must stay pure: remote $ref targets are never
    # fetched, they surface as broken-schema findings instead
    raise NoSuchResource(uri)


_NO_NETWORK_REGISTRY: Registry[Any] = Registry(
    retrieve=_refuse_retrieval  # type: ignore[call-arg]
)


class ToolCallChecker(Checker):
    """Called names must be declared; args must conform to the schema.

    Errors by default: the format spec requires both halves, and a declared
    schema that is not itself a valid JSON Schema breaks the inventory
    (``broken-schema``, checked for every declared tool whether or not it
    is called). The schema dialect is unspecified upstream; an explicit
    ``$schema`` is respected and the default is Draft 2020-12. Schemas
    never trigger network retrieval: an unresolvable reference is a
    ``broken-schema`` finding, not a fetch.
    """

    rules: ClassVar[tuple[Rule, ...]] = (
        Rule(
            id="tool-call/undeclared-name",
            description="a tool call names a declared tool",
            default_severity=Severity.ERROR,
        ),
        Rule(
            id="tool-call/args-invalid",
            description="tool call args conform to the declared schema",
            default_severity=Severity.ERROR,
        ),
        Rule(
            id="tool-call/broken-schema",
            description="every declared tool schema is a valid JSON Schema",
            default_severity=Severity.ERROR,
        ),
    )

    def check(self, conversation: Conversation) -> Iterable[Violation]:
        validators: dict[str, tuple[Validator | None, int]] = {}
        if conversation.system is not None:
            dumped = conversation.system.model_dump(mode="json")
            for index, spec in enumerate(dumped["tools"]):
                name = spec["name"]
                validator, broken = self._compile(spec["json_schema"])
                # The system checker reports duplicate names. This checker
                # still validates every declaration, while calls consistently
                # use the first declaration.
                validators.setdefault(name, (validator, index))
                if broken is not None:
                    yield self.violation(
                        "tool-call/broken-schema",
                        Location(scope="system", field=f"tools[{index}].json_schema"),
                        f"declared schema for {name!r} is invalid: {broken}",
                    )

        for index, item in enumerate(conversation.items):
            if not isinstance(item, ToolCall):
                continue
            if item.name not in validators:
                yield self.violation(
                    "tool-call/undeclared-name",
                    Location(item=index),
                    f"tool {item.name!r} is not declared in the system prompt",
                )
                continue
            validator, declaration_index = validators[item.name]
            if validator is None:
                continue
            args = item.model_dump(mode="json", include={"args"})["args"]
            try:
                errors = [
                    f"{error.json_path}: {error.message}"
                    for error in validator.iter_errors(args)
                ]
            except Exception as error:  # noqa: BLE001
                # e.g. an unresolvable $ref discovered only at validation
                # time; report instead of raising
                validators[item.name] = (None, declaration_index)
                yield self.violation(
                    "tool-call/broken-schema",
                    Location(
                        scope="system",
                        field=f"tools[{declaration_index}].json_schema",
                    ),
                    f"schema for {item.name!r} failed during validation: {error}",
                )
                continue
            for message in errors:
                yield self.violation(
                    "tool-call/args-invalid",
                    Location(item=index, field="args"),
                    message,
                )

    @staticmethod
    def _compile(schema: dict[str, Any]) -> tuple[Validator | None, str | None]:
        try:
            validator_cls = cast(
                type[Validator] | None,
                validator_for(schema, default=cast(type[Validator], None)),
            )
            if validator_cls is None:
                dialect = schema.get("$schema")
                if dialect is not None:
                    return None, f"unsupported JSON Schema dialect {dialect!r}"
                validator_cls = Draft202012Validator
            validator_cls.check_schema(schema)
            validator = validator_cls(schema, registry=_NO_NETWORK_REGISTRY)
        except Exception as error:  # noqa: BLE001
            # any representable-but-broken schema (bad $schema value,
            # invalid keywords, ...) is a finding, never a crash
            message = getattr(error, "message", None)
            return None, str(message) if message else str(error)
        return validator, None
