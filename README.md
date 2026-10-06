# apertus-common

Typed reference library for the text-only Apertus 2 conversation format.
It provides immutable messages, explicit conformance checks, native JSON,
verified token encoding, streaming output parsing and OpenAI message mapping.
The core interface is inspired by `openai-harmony` and `mistral-common`.

**Current state:** the Apertus 2 library migration is implemented. Tokenizer and
framework integration changes are prepared in separate checkouts; model/server
rollout and the message-boundary suspension clarification remain release gates.
Multimodal integration and Jinja maintenance/generation are out of scope.

## Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md): scope, integration/interface plans,
  implementation status, remaining gates and conventions.
- [docs/design.md](docs/design.md): implemented behavior and public contracts.
- [AGENTS.md](AGENTS.md): concise contributor entry point.

The external [format specification](https://github.com/swiss-ai/apertus-2-format-spec-proposal/blob/v1-apertus-2-profile/spec.md)
and [Apertus 2 profile](https://github.com/swiss-ai/apertus-2-format-spec-proposal/tree/v1-apertus-2-profile/profile/apertus_2)
are authoritative. The implemented baseline is revision
`85874b84605f2a0452d53fe5874cc5eddac1b7f4`.
Artifacts belong to [apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer).
The text-only Apertus 2 base/instruct artifacts are proposed in
[PR #35](https://github.com/swiss-ai/apertus-omni-tokenizer/pull/35), independently
of the multimodal extension.
Program tracking: [library #856](https://github.com/swiss-ai/apertus-program/issues/856),
[format #765](https://github.com/swiss-ai/apertus-program/issues/765) and
[integrations #1128](https://github.com/swiss-ai/apertus-program/issues/1128).

## Quick start

```python
from apertus_common import Conversation, Reply, SystemPrompt, User, Wait, render_str

conversation = Conversation(
    system=SystemPrompt.build(behavior="Be brief."),
    items=[User(payload="Hello"), Reply(payload="Hi"), Wait()],
)
report = conversation.check()
assert report.ok and report.complete
assert Conversation.from_json(conversation.to_json()) == conversation
print(render_str(conversation))
```

Constructors validate structure; `.check()` evaluates profile and conversation
rules explicitly. `report.ok` and `report.complete` distinguish errors from
missing evidence. Builders and `.from_json(..., check="full")` provide opt-in
checked construction. Native JSON is versioned and lossless; JSONL datasets
store one conversation per line. The pre-0.2 pilot API and JSON are incompatible.

```python
from apertus_common import from_openai, to_openai

mapped = from_openai([{"role": "user", "content": "Hello"}])
request = to_openai(mapped.value, context=mapped.context)
assert request.value["messages"] == [{"role": "user", "content": "Hello"}]
```

OpenAI mappings return the value, diagnostics and caller-owned tool-ID context.
Unsupported or lossy conversions fail unless explicitly permitted by a mapping
policy. Serving adapters use encoding/parsing; harnesses and dataset tooling
apply the heavier checks.

## Encoding and output parsing

```python
from transformers import AutoTokenizer
from apertus_common import load_encoding

tokenizer = AutoTokenizer.from_pretrained("/path/to/Apertus_2_instruct")
encoding = load_encoding(tokenizer)
prompt_ids = encoding.encode_conversation(conversation)
parser = encoding.output_parser()
# Feed generated token-ID chunks with parser.push_tokens(chunk).
# Drain parser.take_events(); parser.finish() returns a Conversation fragment.
```

Install `apertus-common[hf]` for HF loading. The tokenizer supplies named extra
roles for input/output start/end, header end and wait, plus standard `pad_token`.
The encoder snapshots its backend, preserves payloads, excludes all special IDs
from ordinary text and inserts controls explicitly. No sidecar file is required;
`load_encoding(local_path)` is also supported. Apply
`encoding.generation_constraints` in the engine. Rendering and decoded text are
diagnostic because literal control spellings in payloads are ambiguous there.

```python
from apertus_common import TextStreamParser

parser = TextStreamParser()
parser.push_text("<|out|>reply<|hdr|>Hello<|/out|><|wait|>")
output = parser.finish()
assert output.items == [Reply(payload="Hello"), Wait()]
```

Parsing reconstructs typed messages without checking history, tool declarations
or argument schemas. Call `.check(context=...)` separately.
`StreamParseError` exposes completed output and a valid recovery boundary;
the caller discards the malformed suffix. Pydantic `ValidationError` reports
invalid native structure, `AdapterError` unsupported mappings, `TokenizerError`
invalid artifacts and `ConformanceError` failed checked construction.

## Examples and remaining work

Four executable, type-checked notebooks cover
[construction](examples/core_format_examples.ipynb),
[validation](examples/validation_examples.ipynb),
[datasets](examples/dataset_examples.ipynb) and
[streaming parsing](examples/parsing_examples.ipynb).
The [harness example](examples/harness_roundtrip.py) demonstrates exact token
continuation with and without queued input and malformed-suffix recovery.
Launch notebooks with `uv run --with jupyterlab jupyter lab examples/`.

Remaining rollout tasks are to finalize the owning-repository changes, verify
actual vLLM model generation/HTTP round trips, resolve request suspension versus
logical wait semantics with the specification owners, and publish the release
once these gates pass. Details belong to the
[migration plan](DEVELOPMENT.md#implementation-migration-plan).

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
make test                     # models, checks, mapping, text/token parsing
uv run pytest -k stream        # a focused subset
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
make check-artifact APERTUS_ENCODING=/path/to/artifact  # required real artifact
make harness APERTUS_ENCODING=/path/to/artifact         # token continuation example
make bench APERTUS_ENCODING=/path/to/artifact           # text/token throughput
make build      # build wheel and source distribution
```

Config lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`,
`[tool.pyright]`). Both type checkers stay green: downstream users run
either, and the two disagree in edge cases.
