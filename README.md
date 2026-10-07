# apertus-common

Typed reference library for post-training and serving with the text-only
Apertus 2 conversation format.
It provides immutable messages, explicit conformance checks, native JSON,
verified token encoding, streaming output parsing and OpenAI message mapping.
The core interface is inspired by `openai-harmony` and `mistral-common`.

**Current state:** the Apertus 2 library migration is implemented and verified
against the instruct artifact in tokenizer PR #35. The PR review, framework
adapter fixes, model/server rollout and alignment with the upstream
message-boundary suspension position remain release gates.
Multimodal integration and Jinja maintenance/generation are out of scope.

## Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md): scope, integration/interface plans,
  implementation status, remaining gates and conventions.
- [docs/design.md](docs/design.md): implemented behavior and public contracts.
- [AGENTS.md](AGENTS.md): concise contributor entry point.

The external [format specification](https://github.com/swiss-ai/apertus-2-format-spec-proposal/blob/85874b84605f2a0452d53fe5874cc5eddac1b7f4/spec.md)
and [Apertus 2 profile](https://github.com/swiss-ai/apertus-2-format-spec-proposal/tree/85874b84605f2a0452d53fe5874cc5eddac1b7f4/profile/apertus_2)
are authoritative. The implemented baseline is revision
`85874b84605f2a0452d53fe5874cc5eddac1b7f4`, merged into the spec's `v1` branch.
The library also implements the leading BOS, always-present system `tools`
array and non-empty tool names proposed in
[spec PR #11](https://github.com/swiss-ai/apertus-2-format-spec-proposal/pull/11);
these changes are not yet part of that baseline.
Artifacts belong to [apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer).
The text-only Apertus 2 base/instruct artifacts are proposed in
[PR #35](https://github.com/swiss-ai/apertus-omni-tokenizer/pull/35).
Program tracking: [parent #1129](https://github.com/swiss-ai/apertus-program/issues/1129),
[library #856](https://github.com/swiss-ai/apertus-program/issues/856),
[format #765](https://github.com/swiss-ai/apertus-program/issues/765),
[tokenizer #1139](https://github.com/swiss-ai/apertus-program/issues/1139) and
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
New JSON uses schema v2; `Conversation.from_json()` upgrades unweighted v1 data.

```python
from apertus_common import from_openai, to_openai

mapped = from_openai([{"role": "user", "content": "Hello"}])
request = to_openai(mapped.value, context=mapped.context)
assert request.value["messages"] == [{"role": "user", "content": "Hello"}]
```

OpenAI mappings return the value, diagnostics and caller-owned tool-ID context.
Documented representation changes, such as joining text parts, emit diagnostics;
semantic loss requires an explicit mapping policy. Unsupported conversions fail.
Serving adapters use encoding/parsing; harnesses and dataset tooling
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
roles for input/output start/end, header end and wait, plus standard `pad_token`
and `bos_token`. `encode_conversation` starts with BOS; `encode_items` does not.
Use `encode_items` for output-parser round trips and continuation segments;
the output parser does not accept BOS or input messages.
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

## Data preparation

Start with a dataset-specific conversion to native `Conversation` records. Select
the intended branches, preserve message/tool-result order, choose system settings
and assign tool counters from source evidence. Serialize with `.to_json()`.
Native JSON has `schema_version`, `system` and `items`; OpenAI messages, legacy
`conversation_branches`/`parts` records and rendered control-token text need
explicit conversion. See the [native input example](docs/design.md#native-models-and-storage).

Create the tokenizer and checker tuple once per worker, and pass bounded batches
to `encode_conversations`. Dataset readers, worker pools, reports and resumable
shards belong to the standalone
[posttraining-data](https://github.com/swiss-ai/posttraining-data) integration
in its `apertus2-processing/` directory on `feat/apertus2-data-processing`.
Its README documents prerequisite pipeline stages, JSONL/HF input layouts and
the complete check-then-tokenize workflow. For existing pipeline data, convert
after the recipe's curation/mixture selection and before legacy linearization.
Already-native data needs no numbered legacy stage. Full format checks do not
replace source filtering, decontamination or recipe-specific quality selection.

```python
from apertus_common import checkers_for

checks = checkers_for("full", allow_loss_weights=True)
weighted = (
    Conversation.builder()
    .system(SystemPrompt.build())
    .user("What is 6 times 7?")
    .think("Multiply the numbers.", loss_weight=0)
    .reply("42", loss_weight=1)
    .wait()
    .build()
)
report = weighted.check(checkers=checks)
assert report.ok and report.complete
result = encoding.encode_conversation(weighted, return_loss_weights=True)
assert len(result.token_ids) == len(result.loss_weights)
```

Weights are optional native JSON metadata. Default checking rejects explicit
annotations unless opted in. Missing weights mean output/wait 1 and input/system
0; input envelopes and headers always have weight 0. Explicit output weights
cover the whole frame, and waits have their own independent weight. The returned
vector is aligned to token IDs before any training target shift. Omitting
`return_loss_weights` returns just IDs. Weights never change tokenization.
See the [runnable preparation example](examples/prepare_data.py) and
[weight contract](docs/design.md#rendering-and-encoding).

## Examples and remaining work

Four executable, type-checked notebooks cover
[construction](examples/core_format_examples.ipynb),
[validation](examples/validation_examples.ipynb),
[datasets](examples/dataset_examples.ipynb) and
[streaming parsing](examples/parsing_examples.ipynb).
The [harness example](examples/harness_roundtrip.py) demonstrates exact token
continuation with and without queued input and malformed-suffix recovery.
Launch notebooks with `uv run --with jupyterlab jupyter lab examples/`.

Remaining work: merge tokenizer PR #35 and spec PR #11, finish and pin the
framework adapters, align request suspension with the upstream position,
verify actual vLLM generation/HTTP round trips, then publish.
Details are in [release gates](DEVELOPMENT.md#external-dependencies-and-release-gates)
and [open findings](DEVELOPMENT.md#open-findings-from-the-october-6-review).

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
make prepare APERTUS_ENCODING=/path/to/artifact         # weighted data example
make bench-prepare APERTUS_ENCODING=/path/to/artifact   # preparation throughput
make build      # build wheel and source distribution
```

Config lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`,
`[tool.pyright]`). Both type checkers stay green: downstream users run
either, and the two disagree in edge cases.
