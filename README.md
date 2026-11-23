# apertus-common

Typed reference library for post-training and serving with the text-only
Apertus 2 conversation format. Construct and store conversations, check their
conformance, encode tokens with optional training loss weights, parse generated
output, and map supported OpenAI messages.

The library provides reusable operations; applications own dataset I/O, worker
pools, model generation and tool execution. Multimodal support is out of scope.
For setup, see [Development](#development). Python examples below build on the
same `conversation` and `encoding` objects.

## Table of contents

- [Construct and render conversations](#construct-and-render-conversations)
- [Store and load native JSON](#store-and-load-native-json)
- [Check conformance](#check-conformance)
- [Encode conversations and batches](#encode-conversations-and-batches)
- [Produce training loss weights](#produce-training-loss-weights)
- [Parse generated output](#parse-generated-output)
- [Map OpenAI messages](#map-openai-messages)
- [Serve with vLLM](#serve-with-vllm)
- [Prepare datasets](#prepare-datasets)
- [Documentation and examples](#documentation-and-examples)
- [Development](#development)

## Construct and render conversations

Build immutable typed inputs, outputs and tool interactions. A conversation
stores its system prompt separately from its ordered messages. Constructors
validate structure; conformance checking is explicit.

```python
from apertus_common import Conversation, Reply, SystemPrompt, User, Wait, render_str

conversation = Conversation(
    system=SystemPrompt.build(behavior="Be brief."),
    items=[User(payload="Hello"), Reply(payload="Hi"), Wait()],
)
print(render_str(conversation))
```

Builders offer the same model through chained methods. Rendering is for
inspection: use the encoder for model input because literal control-token
spellings in payloads are ambiguous in rendered text.

Details: [native models](docs/design.md#native-models-and-storage),
[rendering](docs/design.md#rendering-and-encoding),
[construction notebook](examples/core_format_examples.ipynb).

## Store and load native JSON

Native JSON preserves the typed conversation and its training annotations.
Use one serialized conversation per line for native JSONL datasets.

```python
record = conversation.to_json()
restored = Conversation.from_json(record)
assert restored == conversation
```

The native object has `schema_version: 2`, `system` and `items`. Messages use
`type` alone, such as `{"type":"user","payload":"Hello"}`; direction is derived.
Only this schema is accepted, with no migration of older records. OpenAI messages
and legacy dataset records require conversion.

Details: [storage contract and JSON layout](docs/design.md#native-models-and-storage),
[dataset notebook](examples/dataset_examples.ipynb).

## Check conformance

Check profile rules, conversation history and tool argument schemas, returning
structured findings. `ok` means no errors; `complete` means no missing evidence.
Warnings do not fail a report. Reuse checker instances across records to reuse
compiled tool schemas.

```python
from apertus_common import checkers_for

checks = checkers_for("full")
report = conversation.check(checkers=checks)
if not report.ok or not report.complete:
    raise ValueError(f"Conversation did not pass full checking: {report}")
```

Presets are `profile`, `conversation` and `full`. Checked loading through
`Conversation.from_json(record, check="full")` raises `ConformanceError` on
errors or incomplete checks. `CheckContext` supplies evidence for partial
histories; custom checkers can add application rules.

Details: [checking and reports](docs/design.md#explicit-checking),
[validation notebook](examples/validation_examples.ipynb).

## Encode conversations and batches

Load a verified local Apertus 2 tokenizer artifact once per worker. Encode a
single conversation or a bounded batch; batch results match scalar encoding.
Local-path loading requires the `hf` extra, included by `make install`.

```python
from apertus_common import load_encoding

encoding = load_encoding("/path/to/Apertus_2_instruct")
prompt_ids = encoding.encode_conversation(conversation)
batch_ids = encoding.encode_conversations([conversation, restored])
assert batch_ids == [prompt_ids, prompt_ids]
```

`encode_conversation` adds BOS; `encode_items` encodes continuation segments
without BOS. `encode_text` encodes ordinary text. Payloads cannot inject control
IDs. Encoding performs no conformance checks or padding/truncation. Serving
engines must apply `encoding.generation_constraints`.

Details: [encoding contract](docs/design.md#rendering-and-encoding),
[batch examples](examples/dataset_examples.ipynb).

## Produce training loss weights

Optionally annotate messages with `loss_weight` and request a token-aligned
weight vector. Checking these annotations requires explicit opt-in.

```python
weighted = (
    Conversation.builder()
    .system(SystemPrompt.build())
    .user("What is 6 times 7?")
    .think("Multiply the numbers.", loss_weight=0)
    .reply("42", loss_weight=1)
    .wait()
    .build()
)
training_checks = checkers_for("full", allow_loss_weights=True, training=True)
report = weighted.check(checkers=training_checks)
assert report.ok and report.complete
result = encoding.encode_conversation(weighted, return_loss_weights=True)
assert len(result.token_ids) == len(result.loss_weights)
```

The same flag works for `encode_items` and `encode_conversations`. Without it,
encoding returns only IDs. Defaults are output/wait weight 1 and input/system
weight 0; BOS and input framing always have weight 0. Weights do not change
IDs and are returned before any training target shift. `training=True` adds
advisory warnings for input arriving mid-output-burst or a final output without
wait; these valid histories do not fail checking.

Details: [weight defaults and frame boundaries](docs/design.md#rendering-and-encoding),
[training checks](docs/design.md#explicit-checking),
[weighted preparation script](examples/prepare_data.py).

## Parse generated output

Parse generated token IDs into typed outputs, either as a complete segment with
`encoding.parse_output(ids)` or incrementally with an output parser.

```python
output_ids = encoding.encode_items([Reply(payload="Hi"), Wait()])
parser = encoding.output_parser()
parser.push_tokens(output_ids[:1])
parser.push_tokens(output_ids[1:])
events = parser.take_events()
output = parser.finish()
assert output.items == [Reply(payload="Hi"), Wait()]
```

Events expose message starts, payload deltas, completed messages and waits.
The output parser accepts output segments without BOS or input messages.
Parsing does not check conversation history or tool argument schemas.
`StreamParseError` carries completed output and a recovery boundary;
`TextStreamParser` and `parse_output_text` support diagnostic text parsing.

For answer prefills and truncated output, an output message can be `open`
(in memory only): it encodes without its close, and `allow_open=True` keeps an
unfinished last message instead of failing:

```python
from apertus_common import Think

prefill = [Think(payload=""), Reply(payload="The answer is", open=True)]
ids = encoding.encode_items(prefill) + encoding.encode_text(" 4")
output = encoding.parse_output(ids, allow_open=True)
assert output.items[-1] == Reply(payload="The answer is 4", open=True)
```

Details: [parsing and recovery](docs/design.md#output-parsing-and-recovery),
[open output messages](docs/design.md#open-output-messages),
[parsing notebook](examples/parsing_examples.ipynb),
[exact-token harness continuation](examples/harness_roundtrip.py).

## Map OpenAI messages

Convert supported text messages, reasoning and function-tool interactions to
native conversations and back. Mappings return a value, diagnostics and
caller-owned context for preserving tool-call IDs.

```python
from apertus_common import from_openai, to_openai

mapped = from_openai([{"role": "user", "content": "Hello"}])
request = to_openai(mapped.value, context=mapped.context)
assert request.value["messages"] == [{"role": "user", "content": "Hello"}]
```

`to_openai_output` projects an output segment; `OpenAIStreamMapper` maps parser
events to deltas. Supported mappings can normalize representations, such as
joining text parts. Semantic loss requires specific `MappingPolicy` flags;
unsupported conversions raise `AdapterError`. This is a supported subset,
not an exact round-trip guarantee for arbitrary OpenAI or native records.

Details: [supported mappings and loss policies](docs/design.md#openai-mapping),
[format comparison, conversion losses and training/inference examples](examples/openai_conversion_examples.ipynb).

## Serve with vLLM

> [!WARNING]
> The vLLM plugin is temporary work in progress. It will likely be replaced by a
> proper vLLM integration, so its flags, behavior and module may change or be
> removed without notice.

Installing this package next to vLLM (0.28 or newer) registers an `apertus2`
tokenizer mode and parsers through vLLM's `vllm.general_plugins` entry point,
so an Apertus 2 checkpoint serves `/v1/chat/completions` with no chat template:

```sh
vllm serve <checkpoint> --tokenizer-mode apertus2 --reasoning-parser apertus2 \
    --enable-auto-tool-choice --tool-call-parser apertus2
```

The server (and offline `LLM` with this tokenizer mode) refuses to start
without `--reasoning-parser apertus2` (requests would skip the guards below)
and with speculative decoding (vLLM then ignores `logit_bias`). Offline
`LLM(..., reasoning_parser="apertus2").chat()` renders prompts with the plugin
but runs neither the guards nor the parser: set the forbidden-token
`logit_bias` in `SamplingParams` and parse outputs with apertus-common.

Requests:

- Messages are mapped with `from_openai` and encoded to token IDs; leading
  system and developer messages merge into `behavior`, participant `name`s are
  dropped. A later system or developer message is rejected (no host role
  yet).
- Every request gets a -100 `logit_bias` on all forbidden special tokens (a
  strong bias, not a hard mask).
- Rejected with HTTP 400: `ignore_eos`, `min_tokens`, beam search, `stop`,
  stop token IDs other than `<|wait|>`, forced `tool_choice`,
  `parallel_tool_calls: false`, constrained outputs, prompt truncation,
  `allowed_token_ids`, `logit_bias` or `bad_words` on special tokens,
  `reasoning_effort` other than `low`, `medium` or `high` (including `none`,
  `xhigh`, `max`), `enable_thinking: false` and invalid
  conversations such as a trailing assistant message.
- Replay matches the generation: valid tool-call arguments (parsed by vLLM)
  are re-serialized as compact JSON, others replay unchanged. Clients keep a
  burst's order only by sending one assistant message per streamed run.

Responses (streaming and non-streaming alike):

- Generated token IDs are parsed with `TokenStreamParser`; only control token
  IDs frame messages. `think`, `reply` and `call` become `reasoning`, `content`
  and `tool_calls`.
- A call is returned once its message is complete, with its arguments as
  generated and the ID `apertus_<counter>`, so truncated histories keep their
  counters. A counter the model reuses gets a `_` suffix and is renumbered on
  replay (the profile would reject it as a harness error); calls with other
  IDs are numbered after the highest known counter, so mixed histories may
  replay counters out of order.
- Calls are dropped when the request disables tools (no tools, or
  `tool_choice` `none` or `null`). Claims and unknown output
  types are omitted. At malformed output the plugin stops: messages before it
  are returned, the rest is dropped (text already streamed stays).
- Only a cut message is flagged (`finish_reason: length`), and not when a
  complete call came first (vLLM then reports `tool_calls`). The profile's
  recovery needs `/v1/completions` or `return_token_ids`.

Not covered:

- `/v1/completions`: unchanged; clients apply the constraints themselves.
- The batch endpoint: vLLM 0.28 drops the bias there.
- `/v1/messages`: vLLM fails a response whose call arguments are not valid JSON.
- Streamed Responses `function_call` items carry vLLM's own IDs
  (`response.completed` carries `apertus_<counter>`), and with
  `previous_response_id` earlier calls are not seen for counter reuse.

The integration lives in `apertus_common.vllm`, is excluded from the strict
type checks (vLLM has no macOS wheels) and is tested in a CPU-only vLLM
container: `tests/vllm_plugin/Dockerfile`, then `pytest tests/vllm_plugin` with
`APERTUS_ENCODING` pointing at an `Apertus_2_instruct` tokenizer directory. An
end-to-end smoke test serves a tiny random-weight model with the plugin on the
official vLLM CPU image and checks the chat, Responses and Anthropic endpoints:
`tests/vllm_plugin/e2e/run.sh <tokenizer_dir> <model_dir>` (Docker; the model is
created with `make_tiny_model.py`, which needs torch and transformers). CI runs
both in the `vllm/vllm-openai-cpu:v0.28.0` image (job `vllm-plugin`).

## Prepare datasets

The workflow is **curate and select source data → convert to native JSON →
check the complete dataset → tokenize accepted data**. Conversion must choose
source branches, map message/tool-result order, select system settings and
assign tool counters from source evidence. Format checking complements source
filtering, decontamination and quality selection.

For already-native records, the core operations are:

```python
prepared = Conversation.from_json(record, check="full")
prepared_ids = encoding.encode_conversation(prepared)
```

At scale, reuse checkers and the encoding in each worker and submit bounded
batches. Keep whole-dataset checking and tokenization as separate pipeline
steps when checking must finish before any data is tokenized.

The standalone integration in
[posttraining-data](https://github.com/swiss-ai/posttraining-data), directory
`apertus2-processing/` on branch `feat/apertus2-data-processing`, owns dataset
readers, aggregate reports, workers and resumable shards. Its README documents
CLI commands and Parquet/HF input layouts. For legacy pipeline data, convert after
recipe-specific curation/mixture selection and before legacy linearization.
Already-native data needs no numbered legacy stage.

Details: [native input format](docs/design.md#native-models-and-storage),
[dataset notebook](examples/dataset_examples.ipynb),
[runnable per-batch preparation](examples/prepare_data.py).

## Documentation and examples

- [docs/design.md](docs/design.md): implemented architecture and public contracts.
- `DEVELOPMENT.md` (optional, local only): development plans, decisions and
  remaining release gates; excluded from Git.
- [AGENTS.md](AGENTS.md): contributor instructions and reference repositories.
- [Program issue #856](https://github.com/swiss-ai/apertus-program/issues/856):
  library delivery and future considerations.

Five executable, type-checked notebooks cover
[construction](examples/core_format_examples.ipynb),
[validation](examples/validation_examples.ipynb),
[datasets](examples/dataset_examples.ipynb),
[OpenAI conversion and training/inference effects](examples/openai_conversion_examples.ipynb) and
[parsing](examples/parsing_examples.ipynb).
Launch them with `uv run --locked --with jupyterlab jupyter lab examples/`.

The external [format specification](https://github.com/swiss-ai/apertus-2-format-spec-proposal/blob/4f6f170731cfe79c226f3c777204be5e4cb69203/spec.md)
and [Apertus 2 profile](https://github.com/swiss-ai/apertus-2-format-spec-proposal/blob/4f6f170731cfe79c226f3c777204be5e4cb69203/profile/apertus_2/apertus_2.md)
are authoritative; the verified specification baseline is `4f6f170` on `v1`.
The [verified instruct tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer/tree/b801ff1b00d933d10bba90b0c22307157299faa7/tokenizers/Apertus_2_instruct)
is the merged `main` revision `b801ff1` (NFC).

`load_encoding` accepts HF fast tokenizers with a compatible ByteLevel BPE backend,
registered Apertus controls and a distinct BOS token, or a local tokenizer
directory with the `hf` extra. See the [loader requirements](docs/design.md#rendering-and-encoding).
Use an artifact matching your model; loading checks structure, not Git revision,
checksums or model compatibility. External rollout notes stay in local `DEVELOPMENT.md`.

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

No manual venv activation needed: prefix commands with `uv run --locked`, or point
your IDE's interpreter at `.venv/`. Dependencies are managed with
`uv add` / `uv add --dev`, never by editing `pyproject.toml` by hand.

### Tests

```sh
make test                     # models, checks, mapping, text/token parsing
uv run --locked pytest -k stream        # a focused subset
```

### Formatting and checkers

All tooling runs through the `Makefile`; after `make install`, `make all`
runs the same checks as CI. Commands require the committed lockfile to stay
unchanged. Distribution smoke tests additionally resolve consumer dependencies
from the wheel metadata in a temporary environment:

```sh
make            # list targets
make format     # apply formatting and autofixes, including notebooks
make lint       # style checks, including notebooks
make typecheck  # strict mypy + pyright + API completeness
make examples   # type-check and execute notebooks
make rules      # print the conformance rule table
make test       # test suite, including conformance-rule invariants
make all        # everything CI runs
make check-artifact APERTUS_ENCODING=/path/to/artifact  # required real artifact
make harness APERTUS_ENCODING=/path/to/artifact         # token continuation example
make bench APERTUS_ENCODING=/path/to/artifact           # text/token throughput
make prepare APERTUS_ENCODING=/path/to/artifact         # weighted data example
make bench-prepare APERTUS_ENCODING=/path/to/artifact   # preparation throughput
make build      # build wheel and source distribution
make check-dist # build and smoke-test isolated base and HF wheel installs
```

Set `APERTUS_ENCODING=/path/to/Apertus_2_instruct` when running `make examples`
to include the tokenizer-backed encoding sections.

Config lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`,
`[tool.pyright]`). Both type checkers stay green: downstream users run
either, and the two disagree in edge cases.
