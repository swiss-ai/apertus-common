# apertus-common

Typed reference library for the **Apertus 2 conversation format**, following the
`mistral-common` / `openai-harmony` route: one library to validate, parse,
and stringify conversations, so training and serving cannot drift apart.

**Status:** M0, the M1 checker, and the M1 string rendering are
implemented. The remaining M1 work is string parsing (`parse_str`) and
wiring `check()` to strings. Tokens, streaming parsers, and format plugins
follow in later milestones. See [docs/design.md](docs/design.md) for the
architecture and milestone plan.

The token-level format is defined by the
[Apertus 2 format spec](https://github.com/swiss-ai/apertus-2-format-spec-proposal)
(`v1` branch), read-only ground truth for this library.

## Overview

The implemented library has three layers:

- `apertus_common.model` defines immutable Pydantic models for system prompts,
  input and output messages, content parts, and complete `Conversation` values.
  It owns construction, JSON/schema conversion, exact-class round trips,
  message metadata, and explicit input normalization.
- `apertus_common.checker` evaluates relationships across an already valid
  conversation. Its rules cover canonical input order, declared tools and
  arguments, tool-result echoes, waits, ids, and verifiable answers. `check()`
  returns a `ConformanceReport` instead of raising; `dataclasses.asdict()`
  converts it to JSON-ready data.
- `apertus_common.core` renders conversations to the Apertus text format:
  `render_str()` walks the typed model into the wire IR (the middle layer both
  string and token serializations project from) and joins the string
  projection. Control tokens appear as glyphs; payload text is verbatim and
  unescaped, so a payload that spells a control glyph is ambiguous in the
  string form (the token path is unaffected). Parsing that text back into
  models is the next step.

Model validation is local and immediate. It rejects an invalid field or message
at construction time with a Pydantic `ValidationError`, such as a malformed id,
header, media part, or JSON value. Rule checking is conversation-wide and
non-destructive. Individually valid messages remain representable even when
their ordering or relationship is invalid, and `check()` reports those findings
with stable rule ids and severities. `Conversation.normalized()` only provides
the explicit correcting operation for canonical input order.

Keeping model, rule, and codec layers separate makes validation behavior
reusable by training and serving code.

## Usage

### Model and checker

```python
from apertus_common import (
    Assistant, Conversation, Effort, Harness, SystemPrompt, Think,
    TextPart, ToolCall, ToolResult, ToolSpec, User, Wait, type_info,
)

conv = Conversation(
    system=SystemPrompt(
        identity="You are Aria, built by Acme.",
        effort=Effort.MEDIUM,
        tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
    ),
    items=[
        User(payload=[TextPart(text="What should I wear in Lisbon today?")]),
        Think(payload="Need the weather first."),
        ToolCall(id="get_weather:0", name="get_weather", args={"city": "Lisbon"}),
        Wait(),
        ToolResult(id="get_weather:0", payload='{"tempC":19,"cond":"rain"}'),
        Assistant(payload="19°C and rainy; bring a jacket."),
        Wait(),
    ],
)

# Lossless JSON round-trip, exact classes restored:
restored = Conversation.model_validate_json(conv.model_dump_json())
assert restored == conv

# Canonical input order at each boundary (correcting ingest mode):
messy = Conversation(
    items=[User(payload=[TextPart(text="hi")]), Harness(payload="notice")]
)
assert [m.type for m in messy.normalized().items] == ["harness", "user"]

# Order/authority labels are metadata, never serialized:
assert (type_info("in", "user").order, type_info("in", "user").authority) == (5, 3)

# Cross-message conformance is reported, never raised:
from apertus_common import Severity, check

report = check(messy)
assert not report.ok
assert report.errors[0].rule == "input-order/canonical-order"
assert check(messy.normalized()).ok
report = check(messy, overrides={"input-order/canonical-order": Severity.WARNING})
assert report.ok  # loosened to non-breaking

# Render to the Apertus string format: the typed model walks into the
# wire IR (framing as control atoms, text as verbatim runs) and the
# string projection joins it, control tokens as glyphs:
from apertus_common import render_str

print(render_str(messy.normalized()))
# <|in|> type=harness <|hdr|> notice <|/in|>
#
# <|in|> type=user <|hdr|> hi <|/in|>
```

Validated models and their nested sequence/mapping containers are immutable;
construct a new model when changing content so validation cannot be bypassed.

### Interactive notebooks

- [Core format examples](examples/core_format_examples.ipynb): construct every public
  model, inspects metadata and schemas, performs Python and JSON round trips,
  normalizes input order, and demonstrates immutability and validation errors.
- [Validation examples](examples/validation_examples.ipynb): compare local model
  validation with cross-message rules, inspects and serializes reports, changes
  severities, selects checker sets, and defines a custom rule checker.

Launch either notebook without adding Jupyter to the library's runtime
dependencies:

```sh
uv run --with jupyterlab jupyter lab examples/core_format_examples.ipynb
uv run --with jupyterlab jupyter lab examples/validation_examples.ipynb
```

## Development

### Setup

Requires [uv](https://docs.astral.sh/uv/); it manages Python (≥ 3.13),
the virtualenv, and all dependencies.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
git clone git@github.com:swiss-ai/apertus-common.git
cd apertus-common
make install           # locked sync: creates .venv/ and installs dev tools
make test              # verify the setup
```

No manual venv activation needed: prefix commands with `uv run`, or point
your IDE's interpreter at `.venv/`. Dependencies are managed with
`uv add` / `uv add --dev`, never by editing `pyproject.toml` by hand.

### Tests

```sh
make test                     # test suite, incl. an executable copy of the
                              # spec's order/authority table (tests/test_spec_table.py)
uv run pytest -k normalized   # a subset, e.g. just the normalization tests
```

### Formatting and checkers

All tooling runs through the `Makefile`; after `make install`, `make all`
runs the same checks as CI:

```sh
make            # list targets
make format     # apply formatting and autofixes, including notebooks
make lint       # style checks, including notebooks
make typecheck  # strict mypy + pyright + notebook checks + API completeness
make check-repo # repo invariants (unique, well-formed rule ids)
make rules      # print the conformance rule table
make test       # test suite
make all        # everything CI runs
```

Config lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`,
`[tool.pyright]`). Both type checkers stay green: downstream users run
either, and the two disagree in edge cases.
