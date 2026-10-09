# Implemented design

`apertus-common` implements the text-only Apertus 2 profile at specification
revision `85874b84605f2a0452d53fe5874cc5eddac1b7f4`. The external format and
profile are authoritative. Leading BOS, an always-present system `tools` array
and non-empty tool names also follow the decisions proposed in
[spec PR #11](https://github.com/swiss-ai/apertus-2-format-spec-proposal/pull/11),
which are not yet part of this baseline. Scope, integration plans and remaining
release gates are maintained in the optional local-only `DEVELOPMENT.md`.

## Architecture

| Package | Responsibility |
| --- | --- |
| `model` | Immutable typed messages, builders and versioned native JSON |
| `core` | Envelope rendering and the shared output framing machine |
| `checker` | Explicit profile, history and argument checks |
| `encoding` | Verified local tokenizer binding and token-ID parsing |
| `parsing` | Diagnostic text parsing and ordered output events |
| `formats` | Explicit OpenAI request, response and streaming mappings |

Public names are listed in each package's `__all__` and re-exported at the
package root. Internal modules are underscore-prefixed. Encoding loads the
`tokenizers` runtime only when an artifact is opened; argument-schema checking
loads `jsonschema` only when requested. Construction, rendering and parsing do
not run conformance checks. No framework dependency, Jinja template, scheduler,
tool execution or multimodal message model is included. The library targets
post-training and serving; pretraining framing and checking are out of scope.

## Native models and storage

`Conversation(system=None, items=...)` retains supplied order. Its optional
`SystemPrompt` is separate from the items and renders first as a regular system
input carrying compact JSON; `tools` is always present, an empty array when no
tools are declared. Input messages are `Host`, `User`,
`Document(from_, id)` and `Result(name, counter)`; outputs are `Think`, `Reply`,
`Claim` and `Call(name, counter)`. All messages have a text `payload`.
`UnknownOutput(output_type, headers, payload)` preserves unfamiliar output types.
Its native `type` is always `unknown_output`; `output_type` is the original wire
name. All messages expose read-only `.direction` derived from their input/output
class; direction is not a constructor field or serialized metadata.
`Wait` is a separate item. Known types reject extra fields.

`SystemPrompt` contains required `thinking` (`low`, `medium`, `high`), optional
`identity`, `behavior`, `environment`, and tools. Each `ToolSpec` has `name`,
`description`, `json_schema` (serialized as `schema`), and optional `policy`.
`SystemPrompt.build()` defaults thinking to `medium`; direct construction and
native loading require it. `Call.build()` accepts exactly one of raw `payload`
or object `arguments`, serializing the latter as compact JSON. Raw arguments
remain text until explicitly parsed or checked.

Models and their nested collections are immutable. Scalar coercion, non-finite
JSON numbers, invalid Unicode and malformed header values are rejected.
Counters must be integers structurally; negative or out-of-sequence counters
are conformance findings. Header strings cannot contain decoded whitespace,
and document `from` and `id` must be non-empty when present. Tool, `call` and
`result` names must be non-empty; the parser reports an empty call name as an
`INVALID_HEADER` failure.
Payload whitespace and Unicode are preserved, without normalization.

Every model has `builder()`, `from_json()`, `to_json()` and `check()`.
Builders accumulate values through semantic setters and produce immutable
snapshots with `build()`. Conversation builders also provide `user`, `host`,
`document`, `result`, `think`, `reply`, `claim`, `call` and `wait` methods.
There is no separate `from_args` factory or normalization API.

```python
from apertus_common import Conversation, SystemPrompt

conversation = (
    Conversation.builder()
    .system(SystemPrompt.build(behavior="Be brief."))
    .user("Hello")
    .reply("Hi")
    .wait()
    .build(check="full")
)
assert Conversation.from_json(conversation.to_json()) == conversation
```

Native conversation JSON uses `schema_version: 2`; native messages require
their `type` discriminator. Constructors supply fixed defaults.
`Message.from_json()` dispatches to the concrete type. Loading rejects duplicate
JSON keys, constructor-only aliases (`from_`, `json_schema`) and missing or
unsupported versions. Native JSON uses `from` and `schema`. Use these helpers
for storage; Pydantic's lower-level construction/copy APIs are not substitutes for this
contract. `Conversation.model_json_schema()` exports the schema with identifier
`urn:apertus-common:native:2`. The unreleased v2 schema is revised in place:
direction-bearing records and v1 records are rejected, with no migration path.
JSONL datasets contain one native conversation per line. The pilot JSON and
Python API have no compatibility aliases or converters.

Messages, system prompts and waits accept optional `loss_weight` training metadata.
Values are finite non-negative numbers; booleans and numeric strings are rejected.
`None` means unset and is omitted from JSON, while explicit zero is retained.
The metadata never enters headers, payloads or rendered system JSON. Native JSON
round trips preserve it; text/token parsing cannot recover it. Builders expose
the same field, including conversation shortcuts.

A complete native conversation can be stored as the following object. For JSONL,
serialize it onto one physical line with `Conversation.to_json()`:

```json
{
  "schema_version": 2,
  "system": {"thinking": "medium", "tools": []},
  "items": [
    {"type": "user", "payload": "Hello"},
    {"type": "reply", "payload": "Hi"},
    {"type": "wait"}
  ]
}
```

An unknown output uses an explicit storage wrapper:

```json
{"type":"unknown_output","output_type":"quack","headers":{},"payload":"..."}
```

`output_type` follows header type-name syntax and cannot name a known output
(`think`, `reply`, `claim`, `call`). It may be `user`, `wait` or even
`unknown_output` without colliding with native type discriminators. Rendering and
encoding use `output_type`, never the wrapper tag. Profile checking reports
`profile/known-output`; preserving a parsed message does not authorize dispatch.

Input conversion must choose source branches and map content to these typed
messages. Tool calls/results additionally carry top-level `name` and `counter`;
their payloads remain strings. Tool declarations use native `schema`, not the
OpenAI `parameters` key. Dataset IDs and annotations belong outside the native
conversation. Store the JSON string in `conversation_json` for HF, Parquet or
wrapped JSONL datasets; a nested conversation object is a different storage layout.
The standalone pipeline's `apertus2-processing/README.md` documents its supported
Parquet/HF containers and CLI. Encoding this library's models does not establish conformance;
check the selected data explicitly before encoding it.

## Explicit checking

`value.check(mode=None, *, checkers=None, context=None)` returns an immutable
`ConformanceReport`. Conversations default to `full`; other models default to
`profile`. An explicit checker sequence replaces the preset.

| Mode | Checks |
| --- | --- |
| `profile` | Known types, nonnegative counters, document provenance, tool declarations |
| `conversation` | Profile checks plus system presence, call/result history, waits, document IDs and advisory claim placement |
| `full` | Conversation checks plus argument JSON and declared schemas |

`checkers_for(mode, allow_loss_weights=False, training=False)` exposes the presets;
`rule_table()` and `make rules` list all stable rule IDs, including opt-in
training advice. `Checker` supports custom rules. Reports distinguish
`violations` from `unevaluated` checks: `ok` means no errors, and `complete`
means no missing evidence. Warnings do not make `ok` false.
`from_json(..., check=...)` and builders' `build(check=...)` accept a mode or
`CheckOptions`; they raise `ConformanceError(report)` for errors or incomplete
checks. They never return a partially checked success.

Explicit weights produce `profile/loss-weight` errors unless the selected
`ProfileChecker` opts in. Use `checkers_for("full", allow_loss_weights=True)`
for weighted training records. Structural loading validates numbers regardless
of this policy. Context system prompts are checked too; weights do not change
whether two system configurations agree.

Reuse a checker tuple across records in a worker. `ArgumentsChecker` keeps a
bounded process-local LRU cache of compiled schemas and compilation failures,
defaulting to 256 entries; `schema_cache_size=0` disables caching. Keys include
the complete schema, its dialect and property order. Findings and their
locations are generated anew for every record. A fresh preset on every call
cannot reuse this cache. Host checking tracks pending calls incrementally.

`CheckContext` carries complete/partial history, an optional preceding
conversation, checkpoint and explicit host closures. Every document requires
`from` and `id`, and a complete history requires a system prompt.
`HistoryCheckpoint` requires `CheckContext(history="partial", ...)` and supplies
known system, next call counter, call states, document IDs and waiting state
before a retained segment. `None` means unknown;
empty collections mean known empty. Complete histories start at counter zero.
Partial histories evaluate observable facts and report absent evidence as
unevaluated. `HostClosure` identifies a host item's closure of a particular
call; host prose is never interpreted. Reports locate findings in the value,
prefix or checkpoint. No checker mutates these inputs.

Pending calls and unfinished conversations are allowed. After a wait, further
output or another wait requires input.

`training=True` appends `TrainingChecker` to any preset; modes never include it.
It warns about valid histories that turn-based training data rarely intends,
because decoding stops only at a wait: `training/input-mid-burst` for input
directly after output, and `training/final-wait` when the observed history,
including a supplied prefix, ends with output. Standalone values are skipped.
These warnings never change `ok` or `complete`. Result association uses counter and name;
duplicate or unmatched results, reused counters and duplicate document IDs are findings.
Claim-after-reply placement is advisory. Queue scheduling remains a harness
responsibility.

`parse_arguments(call)` returns a fresh mutable dictionary, rejecting duplicate keys,
non-finite values and invalid Unicode with `ArgumentError`.
`check_arguments(call, tool)` reports argument JSON, tool-name and schema
findings without requiring conversation history. Supported
schema dialects are Draft 7, 2019-09 and 2020-12 (default); unresolved/nonlocal
references and unsupported dialects are reported without fetching resources.
Parsing output never invokes these operations.

## Rendering and encoding

The seven controls are `<|in|>`, `<|/in|>`, `<|out|>`, `<|/out|>`, `<|hdr|>`,
`<|wait|>` and `<|pad|>`. An envelope consists of an opening control, type,
optional compact JSON header, header delimiter, payload and closing control.
`render_str(conversation)` inserts no separators or payload whitespace. Rendered
text omits BOS and is diagnostic: literal control spellings inside payloads are
ambiguous.

`load_encoding(tokenizer)` binds a loaded HF fast tokenizer. Six named extra
roles (`input_start_token`, `input_end_token`, `output_start_token`,
`output_end_token`, `header_end_token`, `wait_token`) and standard `pad_token`
identify the seven controls, and the standard `bos_token` names a registered
special token distinct from them. Loading verifies their spellings, IDs and
special registration, the full backend special inventory, and a lossless
BPE/ByteLevel configuration without normalization. It snapshots the backend, disables its
padding, truncation and postprocessor, and never changes the caller's tokenizer.
All backend special IDs are excluded from content, including unnamed reserves;
HF `all_special_ids` alone is insufficient. No binding sidecar is required.
The tokenizer repository's registry and checksums establish artifact provenance.
`load_encoding(local_path)` uses HF locally with the optional `hf` dependency;
artifact acquisition remains the caller's responsibility.

`encode_conversation`, `encode_items` and `encode_text` return token IDs.
`encode_conversation` encodes a complete sequence and starts with
`bos_token_id`, which the tokenizer does not insert itself; `encode_items` and
`encode_text` never add it. Encoding inserts controls explicitly and disables
added-special recognition for all header/payload text, merging adjacent ordinary
text before encoding. It adds no EOS, system, wait or generation prefix and
performs no conformance checks.
Use `encode_items` for output-parser round trips and continuation segments;
the output parser rejects BOS and input messages.
`decode_text` is diagnostic. Token-ID parsing recognizes controls by ID, so
literal spellings in ordinary text remain payload.

`encode_conversations(conversations)` batches ordinary-text runs through the
backend and returns one ID list per conversation, including BOS. Results are
identical to separate scalar calls; an empty batch returns `[]`. The caller
bounds batches and controls worker processes and tokenizer threading.

`encode_conversation`, `encode_items` and `encode_conversations` accept
`return_loss_weights=True`. Scalar operations then return `EncodingResult`
with equally long `token_ids` and `loss_weights` lists; batching returns a list
of those results. IDs are unchanged by weights or the flag. No weight vector
is allocated on the default IDs-only path.

| Region | Default | Explicit annotation |
| --- | --- | --- |
| BOS | 0 | Always 0 |
| Input/system opening, header and closing | 0 | Always 0 |
| Input/system payload | 0 | Message weight |
| Entire output frame, including header and controls | 1 | Message weight |
| Wait | 1 | Wait's own weight, independent of prior output |

These are unshifted floating-point weights: position `i` supervises prediction
of token `i`. Fractional values and values above one are supported. The library
does not shift targets, pad, truncate, pack, schedule jobs or read datasets.

`controls` exposes the seven `ControlTokenIds`. `generation_constraints`
provides wait as EOS and forbids every special ID except output-start,
output-end, header-end and wait. Adapters must apply that mask.
`vocab_size`, `special_token_ids`, `get_vocab`, `token_to_id` and `id_to_token`
provide the metadata needed by framework tokenizer wrappers; `token_bytes`
returns the raw bytes of an ordinary vocabulary entry for incremental decoding.
Request suspension at a message end is distinct from EOS and remains an external
integration gate.

## Output parsing and recovery

`encoding.parse_output(ids)` and `parse_output_text(text)` parse one output
segment into a `Conversation` fragment with no system. Incremental equivalents
are `encoding.output_parser()` (`push_tokens`) and `TextStreamParser`
(`push_text`). Both use the same framing machine; text parsing is diagnostic.

`take_events()` drains ordered `MessageStart(index, type, headers)`,
`PayloadDelta(index, text)`, `MessageCompleted(index, message)` and
`WaitReached()` events. Typed messages are
constructed once, on close. Invalid argument JSON and unknown output types
survive parsing; malformed framing or headers fail immediately.
For unknown outputs, start events and `OpenMessage.type` carry the original wire
name; completed messages store it in `UnknownOutput.output_type`. The harness
does not dispatch them and appends an `UnknownMessageType` host notice at the
next boundary, as required by the Apertus 2 profile. The library does not create
notices or interrupt a well-framed burst.
`MessageStart` guarantees a structurally valid header. Payload deltas remain
provisional until the message closes successfully.

`current` exposes an `OpenMessage`; `boundary` is `EMPTY`, `IN_MESSAGE`,
`MESSAGE_END` or `WAIT`. `consumed` and `last_valid_boundary` use characters for
text and tokens for encoded streams. `finish()` accepts a clean message boundary
without requiring wait, rejects incomplete output and returns the same result
on repeated calls. Any data after wait is an error, including within one push.
A subsequent push after wait or successful finish raises `ParserStateError`;
after failure, both push and finish do so. Events remain drainable.

`StreamParseError` includes reason, offset, unit, message index, last valid
boundary and completed output. Recovery belongs to the caller: discard the
invalid suffix, retain only the original tokens through that boundary, and use
a new parser. Never reconstruct continuation tokens by re-encoding decoded
text. Engine termination reasons are separate from parsing success.

## OpenAI mapping

`from_openai(messages, tools=..., reasoning_effort=..., system_defaults=...,
context=..., policy=...)` constructs native messages and one system prompt.
`to_openai(conversation, ...)` produces a request; `to_openai_output(fragment,
...)` projects an output segment. Each returns `MappingResult(value,
diagnostics, context)`. There is no OpenAI SDK dependency.

`OpenAIMappingContext` explicitly stores external tool IDs and native
counter/name associations, plus the next counter. Retain its returned value
across requests and history compaction. Mapping preserves raw argument strings
and performs no conformance or schema checks. Conflicting settings, unsupported
fields, content kinds or message relationships raise `AdapterError` with a
location and reason.

Supported input includes text system/user/assistant/tool messages, reasoning
text and function tools/calls. Developer messages and multiple system messages
require `MappingPolicy(merge_system_messages=True)` and must precede all other
messages. Text-part arrays are joined with diagnostics,
and an empty assistant message becomes an empty reply with an `empty-assistant`
diagnostic. Null tool fields count as absent; `strict: false` is accepted and
`strict: true` is rejected because no constrained decoding exists.
The default policy rejects mappings that would lose native information,
including host/document/unknown outputs, claim omission, richer system settings
or unrepresentable assistant ordering. `MappingPolicy` explicitly permits
merging system messages, merging assistant messages, omitting claims or
`omit_loss_weights=True`; lossy
conversions produce diagnostics. There is no assistant-prefill support.
Native-to-OpenAI mappings reject annotated records unless weight omission is
explicitly enabled. Native weights supplied through `from_openai(system_defaults=...)`
are preserved on the constructed system prompt.

`OpenAIStreamMapper` consumes the parser's events and returns OpenAI deltas;
`take_diagnostics()` drains mapping notices and `finish()` returns the assembled
assistant message and updated context. Batch projection uses the same mapper.
It validates event ordering and matching completions, but does not reparse
payloads. Streaming deltas remain provisional until completion; a late failure
cannot retract already delivered text. HTTP envelopes, usage accounting and
finish reasons belong to framework adapters.

## Errors and verification

Library exceptions derive from `ApertusFormatError`, except the
`RuntimeError` subclass `ParserStateError`.

| Exception | Contract |
| --- | --- |
| Pydantic `ValidationError` | Invalid native structure, JSON or header values during construction and loading |
| `AdapterError` | Unsupported/conflicting mapping or event sequence |
| `TokenizerError` | Invalid artifact or encoding input |
| `StreamParseError` | Terminal output framing, header or decoding failure; `reason` is a `ParseFailure` |
| `ParserStateError` | Parser reused after termination |
| `ArgumentError` | Invalid argument JSON/object |
| `ConformanceError` | Requested checks failed or were incomplete |

`make all` covers lint, strict typing, exported-type completeness, repository
invariants, all tests and execution/type-checking of four notebooks. Unit token
fixtures are explicitly synthetic. `make check-artifact APERTUS_ENCODING=...`
requires the real artifact and fails if absent. `make bench` reports parser
throughput and memory; timings are observations, not portability guarantees.
`make bench-prepare APERTUS_ENCODING=...` separately measures JSON loading,
checking and scalar/batch encoding, including weighted encoding and host histories.
External framework and model-dependent rollout gates are tracked separately in
DEVELOPMENT.md.
