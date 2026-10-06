"""Strict argument decoding and optional, local-only JSON Schema validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter

from apertus_common.checker._report import (
    ConformanceReport,
    Location,
    Severity,
    Violation,
)
from apertus_common.errors import ArgumentError
from apertus_common.model import Call, ToolSpec
from apertus_common.model._base import strict_json_loads
from apertus_common.model._primitives import FiniteJsonObject

_ARGUMENTS: TypeAdapter[FiniteJsonObject] = TypeAdapter(FiniteJsonObject)

if TYPE_CHECKING:
    from jsonschema.protocols import Validator
    from referencing import Resource
    from referencing._core import Resolver


def _schema_dialect(schema: dict[str, Any]) -> str:
    dialect = schema.get("$schema", "https://json-schema.org/draft/2020-12/schema")
    if isinstance(dialect, str):
        dialect = dialect.removesuffix("#")
    if dialect not in (
        "http://json-schema.org/draft-07/schema",
        "https://json-schema.org/draft-07/schema",
        "https://json-schema.org/draft/2019-09/schema",
        "https://json-schema.org/draft/2020-12/schema",
    ):
        raise ValueError(f"unsupported JSON Schema dialect {dialect!r}")
    return cast(str, dialect)


def parse_arguments(call: Call) -> dict[str, Any]:
    """Decode a fresh object without altering the call's raw payload."""
    try:
        value = strict_json_loads(call.payload)
        # JSON escapes can introduce invalid Unicode after payload validation.
        _ARGUMENTS.validate_python(value)
    except (ValueError, UnicodeError) as exc:
        raise ArgumentError(f"invalid argument JSON: {exc}") from exc
    return cast(dict[str, Any], value)


def _check_references(resource: Resource[Any], resolver: Resolver[Any]) -> None:
    """Resolve only resources registered from this declaration, never fetch."""
    contents = resource.contents
    if isinstance(contents, dict):
        schema = cast(dict[str, Any], contents)
        # Traverse schema resources, not instance data in examples/default/const.
        if "$schema" in schema:
            _schema_dialect(schema)
        for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
            reference = schema.get(keyword)
            if reference is not None:
                if not isinstance(reference, str):
                    raise ValueError("schema references must be strings")
                resolver.lookup(reference)
    for child in resource.subresources():
        _check_references(child, resolver.in_subresource(child))


def _compile_schema(tool: ToolSpec) -> Validator:
    """Compile once per invocation; imports remain outside lightweight paths."""
    from jsonschema import Draft7Validator, Draft201909Validator, Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT7, DRAFT201909, DRAFT202012

    schema = cast(dict[str, Any], tool.model_dump(mode="json", by_alias=True)["schema"])
    dialect = _schema_dialect(schema)
    choices = {
        "http://json-schema.org/draft-07/schema": (Draft7Validator, DRAFT7),
        "https://json-schema.org/draft-07/schema": (Draft7Validator, DRAFT7),
        "https://json-schema.org/draft/2019-09/schema": (
            Draft201909Validator,
            DRAFT201909,
        ),
        "https://json-schema.org/draft/2020-12/schema": (
            Draft202012Validator,
            DRAFT202012,
        ),
    }
    validator_type, specification = choices[dialect]
    validator_type.check_schema(schema)
    resource = Resource.from_contents(schema, default_specification=specification)
    # An empty registry has no network/file retriever. Register only this schema.
    registry: Registry[Any] = (
        Registry[Any]().with_resource("urn:apertus:tool", resource).crawl()
    )
    _check_references(resource, registry.resolver_with_root(resource))
    return validator_type(schema, registry=registry)


def compile_schema(tool: ToolSpec) -> Validator:
    from jsonschema.exceptions import SchemaError
    from referencing.exceptions import Unresolvable

    try:
        return _compile_schema(tool)
    except (
        SchemaError,
        Unresolvable,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
    ) as exc:
        raise ValueError(f"invalid tool schema: {exc}") from exc


def argument_findings(
    call: Call, validator: Validator | None, location: Location
) -> tuple[Violation, ...]:
    try:
        arguments = parse_arguments(call)
    except ArgumentError as exc:
        return (Violation("arguments/json", Severity.ERROR, location, str(exc)),)
    if validator is None:
        return ()
    from referencing.exceptions import Unresolvable

    try:
        return tuple(
            Violation("arguments/schema", Severity.ERROR, location, error.message)
            for error in validator.iter_errors(arguments)
        )
    except (Unresolvable, ValueError, TypeError, RecursionError) as exc:
        # Resolution/recursive-schema failures are declaration failures, not bad args.
        return (
            Violation("arguments/broken-schema", Severity.ERROR, location, str(exc)),
        )


def check_arguments(call: Call, tool: ToolSpec) -> ConformanceReport:
    """Validate JSON and a supplied declaration, without requiring history."""
    findings: list[Violation] = []
    if call.name != tool.name:
        findings.append(
            Violation(
                "arguments/tool-name",
                Severity.ERROR,
                Location(item=0),
                "call and tool declaration names differ",
            )
        )
    try:
        validator = compile_schema(tool)
    except ValueError as exc:
        validator = None
        findings.append(
            Violation(
                "arguments/broken-schema",
                Severity.ERROR,
                Location("system", field="schema"),
                str(exc),
            )
        )
    findings.extend(
        argument_findings(call, validator, Location(item=0, field="payload"))
    )
    return ConformanceReport(tuple(findings))
