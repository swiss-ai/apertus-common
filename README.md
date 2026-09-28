# apertus-common

Typed reference library for the Apertus conversation format, with a core
interface inspired by `openai-harmony` and `mistral-common`. It provides shared
message construction, checking, rendering, and parsing for datasets, training,
evaluation, harnesses, and serving integrations.

**Current state:** a working text-only pilot of the earlier format. Alignment
with the current Apertus 2 profile is still pending. Image/audio message parts
and multimodal integration are not supported.

## Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md): development scope, plans, open decisions,
  and coding conventions.
- [docs/design.md](docs/design.md): implemented architecture and public
  contracts, including native JSON, OpenAI mapping, and streaming parsing.
- [AGENTS.md](AGENTS.md): short entry point for coding agents.

The external [format specification](https://github.com/swiss-ai/apertus-2-format-spec-proposal/blob/v1-apertus-2-profile/spec.md)
and [Apertus 2 profile](https://github.com/swiss-ai/apertus-2-format-spec-proposal/tree/v1-apertus-2-profile/profile/apertus_2)
are the source of truth for format semantics. Tokenizer artifacts are maintained
in [apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer).
Program context: [library and conformance pilot #856](https://github.com/swiss-ai/apertus-program/issues/856)
and [format discussion #765](https://github.com/swiss-ai/apertus-program/issues/765).

## Implemented capabilities

| Package | Capability |
|---|---|
| `model` | Immutable Pydantic models, text parts, JSON Schema, exact-class round trips, explicit input normalization |
| `checker` | Conversation-level rules and structured conformance reports |
| `core` | Shared wire IR and `render_str()` |
| `formats` | Native JSON and OpenAI message mappers; JSONL dataset convention |
| `parsing` | Experimental strict streaming parser for generated output text |

Validation of individual fields happens when models are constructed.
`check()` reports relationships across messages without changing them.
`Conversation.normalized()` explicitly sorts consecutive input messages into
canonical order. Format readers normalize by default; disable that when
auditing observed order.

Native JSON is the library's own lossless serialization. OpenAI conversion
supports a documented text and tool-call subset and rejects unsupported
mappings. Serving integrations can use core encoding and parsing features as
they become available; clients and harnesses use the library's heavier checking
features. The library neither maintains nor generates Jinja templates.

## Planned next tasks

1. Finalize scope, interfaces, and compatibility policy for the pilot API.
2. Align models, headers, system prompts, rendering, checking, and adapters with
   the current Apertus 2 profile.
3. Integrate the tokenizer artifact, add `encode`, and parse output token IDs.
4. Provide reference tool/reasoning consumers and lightweight serving adapters.
5. Package golden conformance fixtures and validate end-to-end integration.

Details and unresolved scope questions live in
[DEVELOPMENT.md](DEVELOPMENT.md#next-work-and-open-decisions).

## Quick start

```python
from apertus_common import (
    Assistant, Conversation, SystemPrompt, User, Wait,
    check, from_json, render_str, to_json,
)

conversation = Conversation(
    system=SystemPrompt(effort="medium", behavior="Be brief."),
    items=[User(payload="Hello"), Assistant(payload="Hi"), Wait()],
)
assert check(conversation).ok
assert from_json(to_json(conversation), normalize=False) == conversation
print(render_str(conversation))
```

String rendering preserves payload text verbatim. Payloads containing control
glyphs are ambiguous in this representation; it is not an untrusted transport
format. The token encoder is not implemented yet.

Dataset files use one native conversation JSON value per line. `from_json()`
and `from_openai()` accept `normalize`, `check`, and `strict` controls:

```python
from apertus_common import from_openai, to_openai

conversation, report = from_openai(
    [{"role": "user", "content": "Hello"}],
    normalize=False,
    check=True,
)
request = to_openai(conversation)
```

`strict=True` raises `ConformanceError` on conformance errors. Invalid model
fields raise Pydantic `ValidationError`; unsupported OpenAI mappings raise
`AdapterError`. See [format contracts](docs/design.md#native-json-and-openai-message-formats).

The experimental `StreamParser` accepts arbitrary text chunks from one output
segment. Drain provisional payload deltas separately from completed messages:

```python
from apertus_common import StreamParser

parser = StreamParser()
parser.push("<|out|> type=assistant <|hdr|> Hello <|/out|>\n<|wait|>")
parser.finish()
assert parser.waiting
messages = parser.take_completed()
deltas = parser.take_deltas()
```

The parser stops at the first failure. `StreamParseError` carries the terminal
parser, prior completed messages, and raw failure data; consumers own recovery.
See the [parser contract](docs/design.md#streaming-output-parser).

## Examples

Four executable notebooks demonstrate the public API:

- [Core format](examples/core_format_examples.ipynb): models, metadata, schema,
  round trips, normalization, and immutability.
- [Validation](examples/validation_examples.ipynb): reports, severity controls,
  and custom checkers.
- [Datasets](examples/dataset_examples.ipynb): JSONL, ingest controls, and OpenAI
  conversion.
- [Parsing](examples/parsing_examples.ipynb): streaming, batch parsing,
  provisional output, failures, and consumer-selected recovery.

Launch a notebook without adding Jupyter as a runtime dependency:

```sh
uv run --with jupyterlab jupyter lab examples/core_format_examples.ipynb
```

## Development

### Setup

Requires [uv](https://docs.astral.sh/uv/); it manages Python (≥ 3.13),
the virtualenv, and all dependencies.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
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
