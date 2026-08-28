# apertus-common

Typed reference library for the **Apertus 2 conversation format**, following the
`mistral-common` / `openai-harmony` route: one library to validate, parse,
and stringify conversations, so training and serving cannot drift apart.

## Progress

- [x] Immutable typed conversation model and system prompt
- [x] Canonical input normalization and conversation conformance checker
- [x] Wire intermediate representation and Apertus string renderer
- [x] Native JSON conversion with `from_json()` and `to_json()`
- [x] JSON Lines dataset convention and examples
- [x] OpenAI chat-completions conversion with `from_openai()` and `to_openai()`
- [x] Experimental strict streaming output parser for generated text
- [ ] Reference tool/reasoning parsers and engine adapters
- [ ] Apertus tokenizer artifact integration and token-ID codecs
- [ ] Packaged golden conformance fixtures
- [ ] Synchronize and validate against the final Apertus 2 format specification
  and tokenizer artifact once they stabilize

The typed model, conformance checker, string renderer, and format adapters are
implemented. Streaming output parsing is experimental. Full transcript parsing
(`parse_str`) remains an optional later extension.

The token-level format is defined by the
[Apertus 2 format spec](https://github.com/swiss-ai/apertus-2-format-spec-proposal)
(`v1` branch), read-only ground truth for this library.

## Overview

The implemented library has five layers:

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
  `render_str()` walks the typed model into the wire IR and joins its string
  projection. The planned token serialization will project from the same IR.
  Control tokens appear as glyphs; payload text is verbatim and unescaped, so a
  payload that spells a control glyph is ambiguous in the string form (the
  planned token path will avoid this ambiguity).
- `apertus_common.parsing` is the experimental serving reverse path:
  `StreamParser`
  parses one contiguous segment of model output back into typed output
  messages, streaming-first and strict (see the parsing interface below).
- `apertus_common.formats` is the completed ingest and export layer.
  `from_json()` / `to_json()` are harmony-style names for the model's own JSON
  (`model_validate_json` / `model_dump_json`, the one canonical and fully
  supported serialization) with the ingest flags `normalize`, `check`
  (return the conformance report too), and `strict` (raise
  `ConformanceError`); `from_openai()` / `to_openai()` map OpenAI
  chat-completions messages to and from the typed model, a deliberately
  lossy adapter with a stated subset and the same ingest flags.

See the [formats guide](docs/formats.md) for the complete native JSON and
OpenAI interfaces, mappings, loss boundaries, and errors.

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

### Datasets: JSON lines to model to string

A dataset file is the JSON-lines convention over the canonical model JSON:
one conversation per line, no library helper needed. `from_json` per line
validates (pydantic `ValidationError` on a bad line), applies canonical
input order by default (`normalize=False` keeps the observed order), and
can return or enforce conformance in the same call:

```python
from apertus_common import ConformanceError, from_json, render_str, to_json

with open("conversations.jsonl", encoding="utf-8") as src:
    conversations = [from_json(line) for line in src]
for conv in conversations:
    print(render_str(conv))

line = to_json(conversations[0])
conv, report = from_json(line, normalize=False, check=True)   # audit observed order
try:
    conv = from_json(line, strict=True)                         # raise on conformance errors
except ConformanceError as error:
    print(error.report.errors[0].rule)

with open("out.jsonl", "w", encoding="utf-8") as dst:
    dst.writelines(to_json(conv) + "\n" for conv in conversations)
```

The line schema is `Conversation.model_json_schema()`, so external tools
can validate a corpus without importing the library.

### OpenAI messages

`from_openai(messages, tools=None, *, effort="medium", continue_final_message=False, normalize=True, check=False, strict=False)`
lifts an OpenAI chat-completions message list into a `Conversation` (same
ingest flags as `from_json`); `to_openai(conversation)` maps back to
`{"messages": [...]}` plus `tools` when declared and `reasoning_effort`
when there is a system prompt. The OpenAI format cannot represent the full
Apertus cast, so the mapping is explicitly lossy:

- in: supported `system`/`developer`, `user`, `assistant`, and `tool` messages
  map into the typed model. A leading `system`/`developer` message becomes
  `SystemPrompt.behavior`
  (later ones become `Harness`), `user` content becomes text/image/audio
  parts, `assistant` becomes `Think` (`reasoning_content`), `Assistant`
  (`content`), and `ToolCall`s (ids byte for byte), `tool` becomes
  `ToolResult`; a `Wait` closes each assistant run (unless
  `continue_final_message=True` for prefill). Accepted messages round-trip
  through `to_openai(from_openai(x))` up to normalization (`developer`
  comes back as `system`, assistant `content` is always present, tool-call
  `arguments` are re-serialized compactly, `name`/unknown keys are dropped,
  and default normalization may move a later `system` notice ahead of the
  messages it followed; `normalize=False` keeps arrival order). Unsupported
  roles, malformed tool calls, and unsupported content-part shapes raise
  `AdapterError`.
- out: only `Harness`, `User`, `ToolResult`, `Think`, `Assistant`,
  `ToolCall`, `Wait`, and a system prompt without `extra_tags` have a
  counterpart; identity, behavior, and environment collapse into one
  `system` string, and a burst not ordered reasoning, content, calls
  becomes several assistant messages (reasoning never enters `content`).
  Anything else (`Retrieval`, `Event`, `Attachment`, `Document`, generic
  types, `VerifiableAnswer`, header `extras`) raises `AdapterError` rather
  than being dropped or relabelled.

```python
from apertus_common import from_openai, render_str, to_openai

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
]
conv = from_openai(messages, effort="high")
print(render_str(conv))
print(to_openai(conv))
```

The same call converts a Hugging Face `datasets` chat corpus (a `messages`
column in chat-completions shape, optionally a `tools` column; Arrow's
`None`-filled keys are treated as absent) into the fully supported form:

```python
from datasets import load_dataset
from apertus_common import from_openai, to_json

rows = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
with open("ultrachat.apertus.jsonl", "w", encoding="utf-8") as sink:
    sink.writelines(to_json(from_openai(row["messages"])) + "\n" for row in rows)
```

A corpus already in the native form travels through `datasets` as text lines
(Arrow cannot type an item `payload` that is a string for most types and a
part list for `user`/`attachment`, so `load_dataset("json", ...)` cannot
infer its schema): use the `text` builder, or a JSON-string column, and hand
the strings to `from_json`:

```python
rows = load_dataset("text", data_files="conversations.jsonl", split="train", streaming=True)
for conv in (from_json(row["text"]) for row in rows):
    ...
```

### Parsing interface

The experimental reverse direction is `StreamParser`, a strict output-only
parser over one contiguous segment of model-generated text. Its API and error
contract may change while tokenizer integration and serving adapters are still
being developed:

```python
from apertus_common import StreamParseError, StreamParser

parser = StreamParser()
try:
    for chunk in engine_stream():
        parser.push(chunk)               # arbitrary text chunk size
        route(parser.take_deltas())      # ordered provisional payload deltas
        commit(parser.take_completed())  # newly validated output messages
    parser.finish()
except StreamParseError as error:
    assert error.parser is parser
    choose_consumer_policy(error.parser, error)

parser.messages                       # cumulative completed messages
parser.waiting                        # wait token accepted

parsed = StreamParser.parse(text)     # batch: one push, then finish
```

The parser accepts only the canonical output grammar and stops at the first
framing, incomplete-input, tool-payload, or model-validation failure. The
structured exception preserves prior state, the raw uncommitted fragment, and
the unconsumed suffix. It also exposes the exact terminal parser as
`error.parser`, including for failures from the batch entry point. It never
selects a recovery strategy: the consumer may retain committed messages, use
or discard provisional deltas, expose raw output, or start a new parser. The
future tool and reasoning parsers consume the same state. `check()` remains
model-only; full transcript parsing (`parse_str`) is an optional later
extension (see [docs/parsing.md](docs/parsing.md)).

### Interactive notebooks

- [Core format examples](examples/core_format_examples.ipynb): constructs every public
  model, inspects metadata and schemas, performs Python and JSON round trips,
  normalizes input order, and demonstrates immutability and validation errors.
- [Validation examples](examples/validation_examples.ipynb): compares local model
  validation with cross-message rules, inspects and serializes reports, changes
  severities, selects checker sets, and defines a custom rule checker.
- [Dataset examples](examples/dataset_examples.ipynb): writes and streams a
  JSON-lines corpus with `to_json`/`from_json`, audits and normalizes it with
  the ingest flags, converts OpenAI messages in and out, and shows what the
  OpenAI mapping cannot carry.
- [Parsing examples](examples/parsing_examples.ipynb): explains the experimental parser
  architecture and interface, demonstrates streaming and batch parsing,
  distinguishes framing, incomplete-input, model-validation, and lifecycle
  errors, inspects raw failure state through `error.parser`, and shows
  consumer-selected recovery with a new parser.

Launch any notebook without adding Jupyter to the library's runtime
dependencies:

```sh
uv run --with jupyterlab jupyter lab examples/core_format_examples.ipynb
uv run --with jupyterlab jupyter lab examples/validation_examples.ipynb
uv run --with jupyterlab jupyter lab examples/dataset_examples.ipynb
uv run --with jupyterlab jupyter lab examples/parsing_examples.ipynb
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
