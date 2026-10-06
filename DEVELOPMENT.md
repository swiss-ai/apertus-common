# Development Planning

This is the single development reference for scope, plans, open decisions, and
conventions. [docs/design.md](docs/design.md) records implemented behavior;
[README.md](README.md) provides the overview and next tasks.

State: The text-only Apertus 2 library migration uses profile revision
`85874b84605f2a0452d53fe5874cc5eddac1b7f4` as its baseline, with the
October 6 BOS and header decisions below anticipating spec PR #11. The
[migration status](#implementation-status) separates verified library behavior
from remaining framework/model rollout gates.

Goal: Build a working implementation of `apertus-common` for dataset
preparation and integration with serving platforms, harnesses, and training
infrastructure.

How: Proceed in three stages, with work in related repositories as needed.

1. Define the high-level scope and the library's role in serving, training,
   and harness integrations.
2. Keep development scope, plans, and conventions in this document. Record
   implemented behavior in `docs/design.md`; use `README.md` for the overview,
   usage, and a short list of next tasks.
3. Define clear interfaces for each capability, adapt the implementation to
   the latest Apertus 2 specification and profile, then review the architecture.

The Apertus 2 tokenizer changes live in a separate owning-repository checkout.
The supported runtime range is Hugging Face `tokenizers>=0.23.2,<0.24`;
verification used `0.23.2`. Encoding, incremental decoding and representative
Apertus workloads are tested. See [implementation status](#implementation-status)
for the prepared changes and remaining rollout gates.

## References

- Local specification repository: `/Users/rkre/swissai_repos/apertus-2-format-spec-proposal`
- Local tokenizer repository: `/Users/rkre/swissai_repos/apertus-omni-tokenizer`
- Core interface reference: [`openai-harmony`](https://github.com/openai/harmony),
  with local checkout at `/Users/rkre/swissai_repos/harmony`.
- Core interface and integration reference:
  [`mistral-common`](https://github.com/mistralai/mistral-common), with local
  checkout at `/Users/rkre/PycharmProjects/mistral-common`.

### Program tracking

The following issues in `swiss-ai/apertus-program` coordinate this work:

- [#1129: Apertus(2) Format and its Integration](https://github.com/swiss-ai/apertus-program/issues/1129)
  is the parent issue coordinating the specification, library, and integrations.
- [#765: Apertus 2 chat template format spec proposal](https://github.com/swiss-ai/apertus-program/issues/765)
  tracks shared format and model-profile decisions.
- [#856: Apertus 2 format: reference library and conformance pilot](https://github.com/swiss-ai/apertus-program/issues/856)
  tracks this library's implementation and the end-to-end conformance pilot.
- [#1128: Integration into training frameworks and data pipelines](https://github.com/swiss-ai/apertus-program/issues/1128)
  tracks the integration inventory, affected components, ownership, and
  dedicated integration sub-issues. Consult it when format or library changes
  require updates in consuming frameworks and pipelines.
- [#1139: Apertus 2 instruct tokenizer build and release](https://github.com/swiss-ai/apertus-program/issues/1139)
  tracks the instruct artifact consumed by `load_encoding`, its review in
  tokenizer PR #35, and token-slot decisions such as BOS/EOS use.

## Stage 1 - Define scope and interfaces

The library provides reusable format operations for Apertus 2 pipelines while
leaving harness responsibilities to its consumers. Target use cases include:

- Post-training (SFT and RL) infrastructure, such as Megatron Bridge and
  NeMo RL. Pretraining data preparation is out of scope.
- Creation and verification of post-training datasets, both raw and tokenized.
- Evaluation in different frameworks.
- Message construction and handling in harnesses and interfaces, such as Pi,
  OpenCode, and Open WebUI.
- Serving integration with Transformers, vLLM, and SGLang.

Named frameworks are target consumers. Dedicated adapters and their ownership
are scoped separately; this list does not commit the repository to maintaining
an integration for every framework.

**Integration design:** Use `openai-harmony` and `mistral-common` as design
references for a clear core interface and lightweight serving integrations.
Serving integrations use only the library's core encoding and streaming
parsing features: they convert prepared messages to tokens or accept token IDs
directly, and expose reasoning, tool calls, and other output types. Heavier
validation and conversation-level conformance checking are provided by the
library and run in client, dataset preparation, or harness code using those
features. Integrations do not duplicate this logic. The parser reports
structural failures; the harness owns recovery, tool execution, and conversation
management.

**Core functionalities**:

- Construct typed messages using Pydantic, enforcing structural validity.
- Check profile and conversation-level conformance separately. Callers may
  require conformance or retain violations for training and evaluation, such
  as calls to undeclared tools.
- Convert typed messages into string representations, native JSON messages,
  or tokens.
- Provide the library's own native JSON message format for serialization and
  dataset storage, with conversion back into typed messages for subsequent
  conversion and checking.
- Provide standard mappers between native typed messages and the OpenAI
  messages format, documenting unsupported mappings and any information loss.

Permitting conformance violations does not bypass structural validation.
Training examples with malformed headers or broken framing require a separate
raw-data path, deferred beyond this migration.

To preserve the format's protection against forged message envelopes, all
text must be encoded with special-token recognition disabled. The encoder
places the message-envelope special tokens first, then tokenizes the text
between them without special-token recognition.

**Additional functionalities and scripts**:

- Parse output token streams or strings back into typed messages.
- Parse messages efficiently at the token level into categories for serving
  integrations such as vLLM and Transformers.
- Provide a reference implementation for fast tokenization of strings or JSON
  messages for serving integrations.

**Integration boundaries:** The library will neither maintain Jinja templates
nor generate them automatically. Integrations use the library's interfaces
directly. Multimodal integration is out of scope for now.

## Technical Integration Plan

The strategy below is implemented in separate tokenizer and framework checkouts.
CPU adapter suites were re-verified against the October 6 tokenizer revision,
as recorded in implementation status. Actual vLLM engine/HTTP continuation and alignment
with the upstream suspension position remain open. Paths, evidence and open
findings are recorded in [implementation status](#implementation-status).

### System prompt creation in both integrations

Transformers and vLLM delegate prompt construction to the model's template or
library; neither universally creates a system prompt. Mistral forwards tools
and reasoning effort through `ChatCompletionRequest.from_openai`.

Both Apertus integrations use one library-owned builder: standing instructions
map to `behavior`, `reasoning_effort` to `thinking`, and tool definitions to
`tools`. Native configuration also exposes `identity`, `environment`, and tool
policies. Construction defaults `thinking` to `medium`; validation of assembled
prompts rejects missing required fields.

Store the system prompt separately and emit it first, exactly once. The harness
chooses and updates its contents; the library rejects duplicate native system
messages and conflicting arguments. Developer instructions or multiple leading
OpenAI system entries require explicit merging policy; later entries are
rejected. For raw-token requests, the harness assembles and validates the prompt
before encoding; vLLM adds no system message. Temperature
and other sampling settings remain engine parameters.

### Transformers: native library backend

Provide a tokenizer backend like `MistralCommonBackend`: `apertus-common` owns
formatting without Jinja; Transformers executes the model.

- **Inputs:** `apply_chat_template` maps `role`/`content` dictionaries; a native
  entry point encodes typed messages or native JSON without loss from chat mapping.
  Both produce token IDs for `model.generate()`. The backend supplies batching,
  padding, attention masks, and tensors. Generic chat pipelines expect
  dictionaries, so native objects require explicit adaptation.
- **Parsing:** expose batch and incremental parsing with ordered native
  messages and a `parse_response` adapter for chat output. Mistral's backend
  provides decoding and inherits generic response-template parsing. Apertus
  adds pipeline wiring that parses original response token IDs, trims padding
  after wait, and preserves tool-ID context. The generic response-template
  path remains separate.
- **Settings:** forward tools and reasoning effort to the library. Continuation
  uses complete-message boundaries; unfinished-message support is deferred.
  Keep `model_max_length` and padding separate from generation settings such as
  EOS and special-token masking.
- **Validation:** Mistral's encoder validates and normalizes using the loader's
  `mode` (default `test`). Apertus requires structural checks; heavier
  conformance checks remain explicit client, dataset, or harness operations.
- **Errors:** follow Mistral's backend: propagate Pydantic/library exceptions
  from construction and encoding with their types and diagnostics; raise
  `ValueError`/`TypeError` for invalid wrapper options.

The chat convention does not constrain `model.generate()`: templates may use
additional fields/roles, and custom backends may expose native APIs. Current
gpt-oss support uses chat dictionaries and a Harmony Jinja template, not
`openai-harmony` or its native message API.

### vLLM: native messages and harness round trips

Start by registering a renderer and parser using the library, following Mistral's
pattern rather than Harmony's dedicated OpenAI endpoint handling. Initially,
the harness manages interleaving through one generation request per output
message, inserting queued inputs before continuation.

- **Renderer:** register an Apertus renderer and tokenizer backend that map
  OpenAI messages and encode prompt IDs through the library, reusing the
  Transformers backend where appropriate.
- **Parser:** wrap the library parser in vLLM's reasoning/tool interfaces with
  one shared state per stream: `think` maps to reasoning, `reply` to content,
  and `call` to tool calls with an explicit counter-to-tool-ID mapping.
  Document unsupported or lossy mappings; native parsing preserves every message
  type and boundary.

Use `/v1/completions` with a token-ID `prompt`, `return_token_ids=True`, and
the `<|/out|>` ID in `stop_token_ids`; `<|wait|>` remains EOS. This bypasses chat
rendering/mapping: the harness encodes and parses through `apertus-common` and
must supply the special-token mask itself, for example through `allowed_token_ids`
containing ordinary IDs plus the four allowed output controls. The chat
renderer/parser request adjustment does not run for raw completions. Closing-token
retention and resubmission still require engine-level verification.
A plugin can later expose native JSON. The spec author's position in
[#765](https://github.com/swiss-ai/apertus-program/issues/765#issuecomment-5233355020)
conflicts with this `<|/out|>` stop; see
[release gates](#external-dependencies-and-release-gates).

Resend the full retained context as token IDs on each request (standard for
vLLM's stateless chat/completions APIs). With previous prompt `P`, returned
tokens `O` including the closing token, and queued inputs `I` encoded in FIFO order:

- Without new input, submit `prompt = P + O` immediately.
- With new input, submit `prompt = P + O + I`.

The parser reconstructs structurally valid typed messages. The harness invokes
`.check()` explicitly where needed and owns tool validation, execution, queues,
and recovery; pending tools do not block continuation. After `<|wait|>`, resume
only with new input.

## Library Interface

The selected interface uses immutable Pydantic models, an
encoding object bound to a tokenizer artifact, and explicit mapping/checking
operations. Signatures below record the selected interface; the decisions and
implementation stages below complete their contracts. Framework wrappers use
this API without adding format logic or harness state.

### Models and construction

| Model | Constructor fields |
| --- | --- |
| `ToolSpec` | `name`, `description`, `json_schema`, optional `policy` |
| `SystemPrompt` | required `thinking: Thinking`; optional `identity`, `behavior`, `environment`; `tools=()` |
| `Conversation` | `schema_version: Literal[1]=1`, `system: SystemPrompt \| None=None`, ordered `items=()` |
| `Host`, `User` | `payload: str` |
| `Document` | `payload: str`, `from_: str \| None=None`, `id: str \| None=None` |
| `Result`, `Call` | `name: str`, `counter: int`, `payload: str` |
| `Think`, `Reply`, `Claim` | `payload: str` |
| `UnknownOutput` | unrecognized `type`, parsed `headers`, raw `payload` |
| `Wait` | no fields; explicit burst boundary |

`Thinking` contains `LOW`, `MEDIUM`, and `HIGH`. Native JSON uses `schema` for
`json_schema`, `from` for `from_`, and message type discriminators. Conversation
items contain input/output messages and waits, never another system prompt.
`schema_version` versions native storage only and is never encoded into tokens.
`system=None` represents a conversation fragment, including parsed output.
Pretraining records are out of scope: every document requires provenance, and
a complete history requires a system prompt. Preserve order
and payloads, including invalid call-argument text and well-framed unknown
output types.

Models support direct constructors and fluent `builder()` construction.
Only `SystemPrompt` and `Call` also expose one-shot `build(...)` factories for
authoring defaults and structured arguments. Builders reuse the same
construction logic and return immutable models:

```python
from apertus_common import Call, Conversation, SystemPrompt, ToolSpec

weather_tool = ToolSpec(
    name="get_weather", description="Weather for a city",
    json_schema={"type": "object", "properties": {"city": {"type": "string"}}},
)
system = SystemPrompt.build(tools=[weather_tool])  # thinking defaults to medium
call = Call.build(name="get_weather", counter=0, arguments={"city": "Zurich"})
same_call = (
    Call.builder()
    .name("get_weather")
    .counter(0)
    .arguments({"city": "Zurich"})
    .build()
)
conversation = (
    Conversation.builder()
    .system(system)
    .user("What is the weather in Zurich?")
    .message(call)
    .wait()
    .build(check="full")
)
```

Message builders expose field setters; `SystemPrompt.builder()` also supports
`.tool(...)`. Conversation builders provide `.system(...)`, `.message(...)`,
and shortcuts for each known message type, including `.call(..., arguments=...)`.
They never infer counters, insert waits, or reorder messages. Reject conflicting
inputs such as both `payload` and `arguments`, or duplicate system assignments.
`Call.build(arguments=...)` serializes a JSON object into the canonical raw
payload; no separate `from_args`/`from_arguments` factory is needed. Construction
defaults belong to builders; direct models and native loading require fields
such as `SystemPrompt.thinking` explicitly.

### JSON and checks

```text
Model.from_json(data: str | bytes, *, check=None) -> Model
Message.from_json(data: str | bytes, *, check=None) -> Message
model.to_json() -> str

conversation.check(mode="full", *, checkers=None, context=None) -> ConformanceReport
message.check(mode="profile", *, checkers=None, context=None) -> ConformanceReport
checkers_for(mode) -> tuple[Checker, ...]
```

Model methods are the canonical API, with no top-level JSON/check wrappers.
They use the native Pydantic schema; standard Pydantic methods remain available.
Loading always validates structure and preserves order. Native JSON accepts
only storage aliases (`from`, `schema`); Python names (`from_`, `json_schema`)
remain constructor/builder arguments. Checking is opt-in and never normalizes
or repairs data.

Modes select cumulative named checker sets:

| Mode | Checks |
| --- | --- |
| `profile` | Known message types, non-negative counters, document provenance, unique tool names and object-schema declaration shape |
| `conversation` | Profile checks plus tool lookup, counters, call/result/host-closure relationships, document-ID uniqueness, system presence, burst transitions and claim-placement guidance |
| `full` | Conversation checks plus schema compilation and strict argument JSON/schema validation |

Explicit `checkers` replace the preset; extend it with
`[*checkers_for("full"), MyChecker()]`. Context supplies the use case and
declarations/history needed by individual checks, separately from the selected
checker set. Missing context is reported as unevaluated, never silently passed.
Pending calls and a clean boundary without `Wait` are valid partial histories.
Severity overrides are deferred until a concrete use case requires them.

`ConformanceReport` exposes `ok` (no error-level violations), `complete` (no
unevaluated applicable checks), `errors`, `warnings`, `violations`, and
unevaluated checks, with rule IDs, severity, locations, and explanations.
`from_json(check=...)` and `.build(check=...)` accept a mode or
`CheckOptions(mode, checkers, context)`, using the same options and checking
logic as `.check()`. They return the model only when `report.ok` and
`report.complete`; otherwise they raise `ConformanceError` carrying the report.
Load first and call `.check()` to inspect the full report. Inapplicable rules
are skipped, separately from checks that lack required context.

Tool validation is also independently callable:

```text
parse_arguments(call: Call) -> dict[str, Any]
check_arguments(call: Call, tool: ToolSpec) -> ConformanceReport
```

The first requires valid JSON with an object root; the second also checks the
declared schema. Neither belongs to framing validation. Business semantics,
tool execution, and selection of the profile's error response remain with
the harness/tool.

### Encoding and engine constraints

```text
load_encoding(hf_tokenizer_or_local_path) -> ApertusEncoding
encoding.encode_conversation(conversation) -> list[int]
encoding.encode_items(items) -> list[int]
encoding.encode_text(text) -> list[int]
encoding.decode_text(token_ids) -> str
render_str(conversation) -> str

encoding.controls -> ControlTokenIds
encoding.bos_token_id -> int
encoding.generation_constraints -> GenerationConstraints
encoding.vocab_size -> int
encoding.special_token_ids -> frozenset[int]
encoding.get_vocab() -> dict[str, int]
encoding.token_to_id(token) -> int | None
encoding.id_to_token(token_id) -> str | None
```

Loading verifies the tokenizer/profile binding, the required control tokens and
a registered BOS token. `encode_conversation` encodes a complete sequence and
starts with BOS, because the base model saw BOS at every pretraining sequence
start and the tokenizer does not insert it.
Encoding enforces structure and faithfully encodes the supplied models, without
running conformance checks or requiring a system prompt. Serving readiness,
including system presence and document provenance, belongs to explicit
`.check()` calls with the relevant context. `encode_items` encodes additional
messages/waits without a system prompt or BOS. Neither method adds EOS, waits or
generation prefixes. Ordinary text cannot encode to special-token IDs; adjacent text
pieces between control tokens are encoded as one run. Text rendering/decoding
is diagnostic, not a lossless transport when literal control-token spellings
occur in payloads.

`ControlTokenIds` exposes `input_start`, `input_end`, `output_start`,
`output_end`, `header_end`, `wait`, and `pad`. `GenerationConstraints` contains
`eos_token_id=wait` and `forbidden_token_ids`: all special IDs except the four
allowed output controls. Ordinary vocabulary remains available. Adapters apply
these constraints; per-request suspension at `output_end` is configured
separately.

### Output parsing

```text
encoding.parse_output(token_ids) -> Conversation
encoding.output_parser() -> TokenStreamParser
parser.push_tokens(token_ids) -> None
parser.take_events() -> list[ParseEvent]
parser.finish() -> Conversation
parser.current -> OpenMessage | None
parser.consumed -> int
parser.boundary -> EMPTY | MESSAGE_END | WAIT | IN_MESSAGE

parse_output_text(text) -> Conversation
TextStreamParser()                  # Same API, with push_text(text).
```

Parsing is lightweight: recognize framing and headers, preserve payload text,
and reconstruct typed messages. It performs only the structural validation
required for those operations, never conformance, tool lookup, counter/history
checks, or argument JSON/schema validation. Batch parsing and `finish()` return
`Conversation(system=None, items=...)` containing the parsed outputs and waits.
The caller explicitly checks that conversation, providing context or assembling
it with the retained history when needed:

```python
output = encoding.parse_output(token_ids)
report = output.check(mode="full", context=check_context)
```

Batch and incremental parsing share one framing machine. Each parser handles
one contiguous output segment and emits one ordered event stream:

| Event | Fields and meaning |
| --- | --- |
| `MessageStart` | `index`, `type`, `headers`; header structurally valid |
| `PayloadDelta` | `index`, `text`; provisional payload fragment |
| `MessageCompleted` | `index`, `message`; structurally complete native message |
| `WaitReached` | burst ended |

`OpenMessage` exposes the current index, type, headers, and accumulated payload.
Parser progress stays on the parser, separate from the returned conversation;
`boundary` is `IN_MESSAGE` while a frame is open. Offsets/counts identify their
unit: tokens for token parsing, characters for text parsing. `finish()` accepts
a clean boundary without a wait; incomplete or malformed framing raises a
terminal parser error. Engine termination reasons remain separate. The parser
collects typed messages for the returned conversation; it does not retain a
separate raw transcript or conformance state.

A completed `Call` retains its raw payload even when its arguments are invalid.
Well-framed unknown types produce `UnknownOutput`. The harness handles both
according to the profile; malformed framing discards the invalid suffix through
harness recovery. Full-transcript parsing is not included in this initial
surface.

### OpenAI mapping

```text
from_openai(messages, *, tools=None, reasoning_effort=None,
            system_defaults=None, context=None, policy=None) -> MappingResult[Conversation]
to_openai(conversation, *, context=None, policy=None) -> MappingResult[OpenAIRequest]
to_openai_output(output: Conversation, *, context=None, policy=None)
    -> MappingResult[OpenAIAssistantMessage]

OpenAIStreamMapper(context=None, policy=None)
mapper.push(event: ParseEvent) -> Sequence[OpenAIDelta]
mapper.take_diagnostics() -> list[MappingDiagnostic]
mapper.finish() -> MappingResult[OpenAIAssistantMessage]
```

`MappingResult[T]` always contains `value`, `diagnostics`, and an updated
`OpenAIMappingContext`. Context holds serializable OpenAI tool-ID/native
counter-name associations, retained by the caller across requests and history
compaction. No global mapping state is hidden in the library. Diagnostics
identify lossy conversions. Conflicting settings and unsupported conversions
raise `AdapterError`; only the specific lossy conversions enabled by
`MappingPolicy` are permitted. The default policy enables none.

Input mapping uses the shared system builder; output mapping consumes parsed
messages/events without reparsing tokens. `to_openai_output` accepts a
conversation fragment containing only outputs/waits. Unfinished-message
continuation is deferred; wrappers reject requests for assistant prefilling.
Framework adapters own HTTP response envelopes, usage accounting, and engine
finish reasons.

### Error contract

| Error | Meaning |
| --- | --- |
| `pydantic.ValidationError` | Invalid native model/JSON structure |
| `AdapterError` | Unsupported or conflicting external-format mapping |
| `TokenizerError` | Invalid artifact/binding or encoding failure |
| `StreamParseError` | Malformed framing/header or incomplete output |
| `ArgumentError` | Invalid argument JSON or non-object root |
| `ConformanceError` | Requested checks failed; carries the report |
| `ParserStateError` | Parser used after termination or otherwise misused |

Parser errors expose a stable reason code, offset/unit, message index, last
valid boundary, and completed output. Recovery uses a new parser selected by
the harness. The library preserves exception types and diagnostics. Framework
adapters may translate them at their API boundary: Transformers propagates them,
while vLLM wraps library failures in `ValueError` for HTTP 400 responses.
No automatic recovery, execution, or scheduling is added by the library.

### Decisions for the first implementation

These choices close the interface questions for this migration. They are library
policies where the profile leaves a choice, not changes to the upstream format.

**Models and storage.** Reject unknown fields on known types, invalid Unicode,
non-finite JSON numbers, and scalar coercion (including booleans as counters).
Accept Python sequences/mappings and freeze them recursively. Counters must be
integers structurally; non-negativity, uniqueness and sequencing are checks.
Known messages have only their declared header fields. Header syntax, including
the ban on whitespace in decoded header strings, is structural and always
enforced by models/parsing. Document `from` and `id` are non-empty when present
(the profile's "required, non-empty strings"). Tool, `call` and `result` names
are non-empty: the profile requires a declared tool `name` that the call copies
and the harness dispatches on, so an empty name is a malformed header. Spec
PR #11 states this explicitly. Preserve unknown output
types and their syntactically valid compact JSON headers as `UnknownOutput`;
do not invent unknown input types. `Message` is the common public base;
`Message.from_json()` dispatches using native `direction` and `type`, while
`Wait` remains a separate conversation item. Direct constructors supply fixed
discriminators; native message loading requires them. Strict JSON helpers reject
duplicate keys and non-finite values with Pydantic `ValidationError` diagnostics.

Release as `0.2.0`, with no aliases or automatic conversion for the pilot API,
old wire grammar, or old JSON. Conversation JSON requires `schema_version: 1`;
constructors supply it, serializers emit it, and unsupported/missing versions
fail loading. Standalone messages use the same versioned schema definition
without repeating a version field on every item. Export the JSON Schema with a
versioned identifier; future incompatible storage changes increment the version.

**Checking context.** Use immutable
`CheckContext(history="complete", prefix=None, checkpoint=None,
host_closures=None)`. `history` is `complete` or `partial`, describing whether the supplied prefix
and value cover the history from its origin, not whether generation has ended.
`prefix` is a conversation immediately preceding the checked value. A
`HistoryCheckpoint` supplies known system configuration, next call counter,
call states (name and pending/result/host-closed status), document IDs, and
waiting state before that prefix/value. Unknown checkpoint fields stay unknown;
empty collections mean known empty state. It is a caller-owned snapshot, never
a mutable library session or something the checker updates implicitly.

For a complete history, begin at counter zero with no earlier calls/documents.
For a partial history, evaluate observable local facts and mark unavailable
historical checks unevaluated. Compaction that removes interior calls cannot be
reconstructed from a prefix checkpoint: check retained contiguous segments with
their own checkpoints or report those historical checks unevaluated. A
`HostClosure` identifies the host item and affected call counter/name explicitly;
the library never interprets host prose. `host_closures=None` means unknown
closure metadata; an explicit empty tuple declares that no host items close
calls. With host messages and relevant calls, unavailable closure evidence
makes affected relationship checks unevaluated. Locations distinguish checked
items from prefix items. Standalone message checks without context evaluate local
rules and mark any applicable context-dependent checks unevaluated.

An output fragment can be checked with
`CheckContext(prefix=prompt_conversation)`. A missing fragment system is not a
second missing-system error when the prefix/checkpoint supplies it. Contradictory
context is a caller error. Pending calls, bare waits, and a tail without a wait
are valid partial histories; no generic end-of-transcript warning is emitted.
For observed complete transitions, output following a wait requires intervening
input. Queue arrival order cannot be verified from messages alone: preserve
order, remove the old input-sorting rule, and leave scheduling to the harness.
The profile places a claim before the reply it supports; retain an advisory
placement check for an observed claim after a reply in the same burst. Do not
infer one task per burst or impose a one-claim-per-burst limit. Normative
requirements are errors; this placement heuristic and custom advisory checks
are warnings. `ok` and `complete` are independent, as defined above.

**Tool validation.** Default to JSON Schema Draft 2020-12; honor explicit
Draft 7, 2019-09, or 2020-12 dialects and report unsupported dialects. Permit
references within the same schema, with no file/network resolution. Validate
every declared schema, even unused ones; report broken schemas separately from
invalid arguments. `format` remains an annotation, not an implicit format
checker. Arguments require an object root, reject duplicate keys at every
depth and non-finite numbers, and receive no coercion/default insertion. Keep
the raw call payload unchanged. Compile each schema once per check invocation;
the parser and encoder never load or invoke schema validators. Business semantics
and tool execution remain outside the library.

**Parser lifecycle and round trips.** Parse one contiguous output segment,
without inter-message separator whitespace or implicit payload trimming. Batch
parsing and `finish()` return a conversation fragment. `Wait` ends accepted
input; trailing data in the same push raises `TRAILING_AFTER_WAIT`, and another
push raises `ParserStateError`. `finish()` after a wait or clean boundary
succeeds and is idempotent, returning the same immutable result. Finishing an
incomplete frame raises `StreamParseError`; failure is terminal. Events may be
drained after success or failure. `current` is available after a valid header;
`consumed` counts accepted input units, while errors also identify the last
complete boundary for slicing the caller's original tokens.

Retain completed typed messages once, plus the current frame and undrained
events; no second raw transcript, conformance state, or retention-mode option.
Memory is proportional to the returned output segment; harness round trips
naturally bound each segment. Defer an event-only parser until demonstrated
need. Token decoding must handle Unicode split across chunks without replacement
characters or repeated decoding of the entire prefix; an undecodable sequence
at a message boundary is a parser failure, not silently repaired text.

Native JSON round trips preserve model values, order and raw payloads, not the
input JSON's whitespace. For a single representable output segment, encoding
followed by token parsing preserves typed messages and payloads; accepting a
tokenizer requires proving its ordinary-text path preserves those payloads.
Parsing arbitrary generated tokens and re-encoding does not promise identical IDs or header key
ordering. Keep original tokens for generation continuation. Text parsing has
the documented control-glyph ambiguity and is only a diagnostic frontend.

**OpenAI compatibility.** Initially support text `system`/`developer`, `user`,
`assistant`, function tools/calls and tool results; `reasoning_content` is an
explicit compatibility extension. No OpenAI SDK dependency: use library-owned
typed dictionaries. Preserve raw argument strings, require tool-call IDs, and
reject unsupported roles, meaningful unknown fields, non-text content and
ambiguous mappings. Text-part lists may be joined with a conversion diagnostic.
Missing or null tool descriptions become empty strings; absent or null
parameters use an empty object schema. `strict: false` is accepted; `strict:
true` is rejected because there is no constrained decoding. An assistant
message without content, reasoning or calls becomes an empty reply with an
`empty-assistant` diagnostic. Do not turn missing/malformed call arguments into `{}`.
These representation changes are documented normalizations; semantic loss
requires a specific policy flag and is never covered by a generic allow flag.

One leading `system` becomes `behavior`. A `developer` entry or multiple leading
system/developer entries require `MappingPolicy(merge_system_messages=True)`;
join them in order with blank lines and report lost role/priority distinctions.
Later entries are rejected rather than relabelled as host messages. Explicit
tools/effort and `system_defaults` must agree where both specify a value; only
unspecified values receive defaults.
Non-default identity/environment/tool-policy data cannot be silently flattened
when mapping back to OpenAI: reject it in this initial subset.

Assign new call counters in encounter order, reuse existing associations on
history replay, and retain original OpenAI IDs in `OpenAIMappingContext`.
Context also carries the next counter; compacted input needs that context.
Native-to-OpenAI mapping allocates deterministic, collision-checked IDs when
needed. Reject unresolved results or conflicting ID/name/counter associations.

An OpenAI assistant record maps in reasoning, content, tool-call order; a
completed assistant run ends with `Wait`. Native-to-OpenAI history conversion
splits records as needed to preserve supported ordering; if burst/message
boundaries cannot survive, require the merging policy there as well. A single
response or stream cannot preserve every native message boundary: adapters use
`MappingPolicy(merge_assistant_messages=True)` to concatenate reasoning/reply
segments and collect calls, reporting lost boundaries/order. An additional
`omit_claims=True` permits ignoring claims with diagnostics. All policy flags
default false; host, document, unknown types and other unsupported conversions
still fail. Streaming diagnostics are drainable immediately and included in
the final result; streaming failure cannot retract previously emitted deltas.
The mapper never dispatches a tool or runs conformance checks.

**Initial scope.** Support complete-message continuation only. Reject wrapper
options requesting unfinished assistant prefixes; add no implicit generation
prefix. Defer full-transcript parsing, malformed/raw training-data ingestion,
multimodal integration and native HTTP endpoints. Keep frame/error handling,
conformance and OpenAI mapping separate; do not add a session manager.

## Implementation Migration Plan

The following sequence records the agreed migration plan. Progress and remaining
gates are summarized below; it is not a claim that framework rollout is complete.
Baseline:
library commit `a23cced` and profile repository commit `85874b8`, inspected
October 1, 2026. The tokenizer working tree is on `main` at `a79ba0a`; the
profile-referenced Apertus 2 candidate is at `ba648ff` on `v2-on-pr24`, inspected
through Git without switching branches. Changes to the source profile during
implementation must be reconciled explicitly.

### Original implementation and reuse

| Area | Retain | Replace or add |
| --- | --- | --- |
| `model/` | Recursive immutability, strict primitives, typed dispatch | Apertus 2 types, native JSON methods, builders; remove text-part payloads, generic inputs, header extras on known types and sorting |
| `core/` | Private control/text atoms and source spans | Seven controls, compact JSON headers/system and exact payload rendering |
| New `encoding/` | Renderer atoms and lexer protocol | Artifact-bound encoding, ordinary-text backend and token frontend |
| `parsing/` | Lexer/machine separation and terminal failure handling | New grammar, token frontend, ordered events, conversation result; remove argument parsing and duplicate raw buffering |
| `checker/` | Rule IDs, locations, reports, schema-validation utilities | Presets, explicit history context and unevaluated results; new counters, provenance and closure rules |
| `formats/` | Supported conversion cases and failure locations | Strict OpenAI mapping, persistent caller-owned ID context, batch/stream output mapping; retire top-level JSON helpers |
| Tests/scripts/docs | Immutability, chunk-split and negative-test patterns, Makefile gates | Current profile fixtures, new public API, four notebooks, rule tools and integration tests |

Map `Harness` to `Host`, `Assistant` to `Reply`, `VerifiableAnswer` to `Claim`,
and `Effort` to `Thinking`. `ToolCall`/`ToolResult` become `Call`/`Result` with
separate name/counter headers and raw payloads. Attachment/retrieval data needs
explicit `Document.from_`/`id`; this is not a mechanical rename. Remove the old
`NAME:COUNTER` grammar, `<|sys|>` controls, XML system tags, `type=...` headers,
structural payload spaces and newline separators.

### Delivery sequence

Use reviewable commits, including affected tests, examples, exports and
implemented-design documentation. Stages 2-4 are a coordinated native-API
cutover: keep intermediate changes on the migration branch and integrate the
dependent model, checker and parser changes before calling that cutover ready.
Temporary internal scaffolding must not become compatibility API. Run the
existing `make all` once before code changes to establish the baseline. During
migration run focused tests and relevant Makefile checks; require the full gate
at each coherent integration checkpoint and before release, without disabling
old coverage merely to make intermediate commits green.

1. **Pin contracts and fixtures.** Create a profile-to-rule matrix classifying
   framing/model invariants, optional checks, mapping rules and harness duties.
   Record the source revision and literal expected wire examples independently
   of the renderer. Cover system JSON, all message types, calls/results,
   pretraining, FIFO interleaving, waits and malformed output. Export snapshots
   of the target native JSON schema and public signatures as the implementation
   lands; do not introduce skipped tests as evidence of completion.
   **Exit:** every normative requirement has an owner and acceptance case;
   known source conflicts are recorded below. Pretraining coverage in the
   original plan was removed by the October 6 scope decision.

2. **Replace models, construction, JSON and string framing.** Update
   `model/_base.py`, `_primitives.py`, `_system.py`, `_input.py`, `_output.py`
   and `_conversation.py`; add private builder/JSON helpers. Implement strict
   `Message.from_json` dispatch and versioned conversation loading. Rewrite
   `core/_wire.py`/`_render.py` to render one input system message with compact
   JSON and concatenate exact envelopes without injected whitespace. Remove
   normalization and old public exports. Port dependent modules to these types
   across the native cutover; expose checking factories only with stage 3's
   working checker delegation, not placeholder success.
   **Exit:** constructor/builder equivalence, immutable nested values, schema
   validity, native round trips, literal render fixtures and rejection of old
   JSON/grammar all pass.

3. **Implement explicit checking.** Rework `_check.py`, `_report.py` and rule
   families for presets, context/checkpoints, closures and report completeness.
   Split strict argument parsing from schema checking. Add model `.check()`
   delegation and checked loading/building through one options path, using lazy
   imports to avoid model/checker cycles and load `jsonschema` only for schema
   checks. Retire input-order and old ID rules; update `scripts/rules.py`,
   `scripts/check_repo.py` and validation examples.
   **Exit:** partial histories do not invent errors, missing evidence is
   unevaluated, complete histories catch actual violations, schema failures are
   distinguished from argument failures, and input objects never change.

4. **Replace output parsing.** Rewrite the grammar in `core/_machine.py`,
   adapt the text lexer/facade, and introduce the common ordered events and
   conversation return. Construct each typed message once at close; retain
   invalid argument text and unknown output messages without conformance calls.
   Implement the selected lifecycle, offsets, completed-output error recovery
   and no raw transcript copy. Keep control/token frontend concerns separate.
   **Exit:** batch/stream results match across every split of small fixtures
   and seeded large splits; whitespace and Unicode payloads survive; malformed
   headers fail while invalid arguments parse. Regression tests make checker
   and schema-validator entry points fail if invoked by parsing.

5. **Bind the tokenizer and add token parsing.** Add private artifact loading,
   ordinary-text encoding and incremental decoding in `encoding/`, behind
   `ApertusEncoding`.
   Reuse renderer atoms and the output machine; merge adjacent ordinary text
   before tokenizing. Recognize controls by token ID, never by scanning decoded
   payload text. Expose controls and generation constraints from the
   verified artifact, with no hard-coded production IDs. Load from an explicit
   HF fast tokenizer or local artifact directory; artifact acquisition belongs
   to the caller. Keep Transformers optional (`hf` extra) and vLLM external.
   **Exit:** real-artifact golden IDs, text/token parser equivalence on the
   unambiguous subset, split Unicode decoding, literal special-token spellings
   as ordinary text, model-value round trips and complete special-ID masking
   pass. Test doubles may exercise the machine, but cannot satisfy this gate.

6. **Replace OpenAI mapping and test harness round trips.** Rework
   `formats/_openai.py` around typed results, policies and explicit ID context;
   add the output/event mapper without a second parser. Exercise complete
   history replay, compacted history with context, parallel/out-of-order
   results, policy diagnostics and late streaming errors. Add a small harness
   example that preserves `P + O` or `P + O + I` and checks returned messages
   explicitly before execution decisions.
   **Exit:** batch and streamed projections agree for the supported subset,
   replay preserves IDs/counters, no unsupported data disappears silently,
   and a malformed suffix is never dispatched or included in continuation.

7. **Integrate and release.** In the owning integration repositories, implement
   the Transformers backend/response adapter and vLLM renderer plus shared
   reasoning/tool parser. Register them through supported extension points;
   add no Jinja. Pin tested framework revisions and exercise real tokenizer,
   backend batching/padding, EOS/masking, native messages and OpenAI mapping.
   Verify vLLM returns the closing token IDs, accepts the concatenated prompt,
   and resumes at message boundaries with and without queued input. Verify a
   wait requires input and cannot be resumed solely by a pending call.
   **Exit:** framework smoke tests plus the library's `make all` pass; all four
   notebooks execute/type-check; README, `docs/design.md`, exports and installed
   wheel behavior match the implementation. Publish migration notes with the
   `0.2.0` release; update implemented docs incrementally, not in advance.

Stages 2-4 establish the native text path. Updating the existing Apertus 2
tokenizer recipe in its own repository can proceed alongside them; stage 5
consumes the verified artifact. Stage 6's message mapping can begin after
stages 2-4, but its harness token test requires stage 5. Stage 7 depends on
stages 5-6; library completion and framework rollout
are reported separately, never inferred from mocked integration tests.

### Implementation status

The `0.2.0` native cutover is implemented without pilot compatibility aliases.
Models, builders, JSON, rendering, explicit checking, text/token parsing and
OpenAI mapping use the interface above. Parser reconstruction remains independent
of conformance and argument validation. All four notebooks use the new API;
[implemented contracts](docs/design.md) and README describe it.

| Profile responsibility | Owner and acceptance coverage |
| --- | --- |
| Seven controls, compact envelopes, system JSON, exact payloads | Models/core; literal fixtures in `test_render.py`, inventory in `test_spec_table.py` |
| Typed headers, discriminators, immutable native JSON | Models; header/JSON/immutability tests, versioned schema snapshot in `tests/fixtures/native-v1.schema.json` |
| Document provenance, unique tools/documents, call/result counters | Explicit profile/conversation checkers; `test_checker.py` and `test_checker_rules.py` |
| FIFO arrivals, queued-input placement, dispatch and host closures | Harness; order preserved by models/encoding, explicit closure evidence in checks, `examples/harness_roundtrip.py` |
| Argument object syntax and declared schema | Explicit argument helpers/checker; malformed calls retained by both parsers, checker regressions |
| Framing errors, unknown outputs, wait boundary and incomplete generation | Shared parser; exhaustive small splits, failure/reconstruction suites and token tests |
| Literal control text, Unicode bytes, header tokens, EOS and special-ID mask | Verified tokenizer binding; synthetic unit tests plus required real-artifact golden-ID check |
| OpenAI subset, ordering/loss policy, tool-ID continuity | Mappers; batch/stream tests and explicit context, no implicit normalization |
| Engine suspension/resume and HTTP/tool execution | Framework/harness; CPU adapter tests implemented, actual vLLM engine gate still pending |

Independent review iterations corrected incomplete-check reporting, unsupported
nested schema dialects, argument Unicode validation, native-schema constraints,
mapper error handling, tokenizer configuration validation and framework request
plumbing. Reviews and executable checks are required again after any further
contract change.

Integration checkouts and the tokenizer symlink are under
`/Users/rkre/swissai_repos/apertus2-integration-work/`:

- `tokenizer-text`: a symlink to `/Users/rkre/swissai_repos/apertus-omni-tokenizer`,
  on branch `feat/apertus-2-text-tokenizers` at `d00a5c1`. The duplicate checkout
  was consolidated on October 6, preserving local files, including `future.md`,
  and backup branches. `tokenizers/Apertus_2` is the byte-identical
  `preliminary_mul_200k` source from `swiss-ai/apertus-tokenizer-development`
  revision `28ad57a2757f6f72edb0def57ee725b6812f2df3`; its tokenizer SHA-256 is
  `cd403d3f219e2433e3f78b32644b8e6a6134668e15138e6546360330635a96b9`.
  `tokenizers/Apertus_2_instruct` exposes named HF control roles at IDs
  7/8 (input), 10/11 (output), 9 (header), 12 (wait), and 3 (pad), and disables
  text rewriting/automatic BOS/EOS. No binding manifest is needed. Both have
  200,064 tokens, unchanged BPE merges and no appended media vocabulary.
  Compared with the base, instruct moves the PII tokens from IDs 24-26 to 4-6,
  drops the NFC normalizer and declares `generation_config.json` EOS `[12, 2]`
  (`<|wait|>`, `</s>`). Deterministic rebuild, provenance and deployment tests
  accompany the artifacts; generation settings are validated without
  overwriting checkpoint-specific sampling defaults.
- `tokenizer`: retained historical prototype on PR #34's multimodal branch
  `v2-on-pr24` (`ba648ff`). PR #34 was closed unmerged on October 2 because
  multimodal is not planned for Apertus 2. Its 335,232-token interaction
  artifact is superseded by `tokenizer-text/tokenizers/Apertus_2_instruct`.
- `transformers` at `166bf9ea17e6052be11680cd3ae888d103bcddf6`:
  `ApertusCommonBackend`, native/token response handling and eight CPU
  integration tests. It exposes native conversations and OpenAI mapping
  without Jinja or implicit conformance checks. It loads the fast tokenizer
  directly and saves the artifact's named roles, so `AutoTokenizer` reloads
  saved directories without the removed `apertus_encoding.json` sidecar.
- `vllm` at `806a65923b6c3ced8d933abfbd744834ca7bad88`:
  registered tokenizer/renderer and unified reasoning/tool parser,
  token-ID projection and request masking. CPU tests exercise actual framework
  request, renderer and parser classes. GPU/HTTP generation is not verified.

The original tokenizer checkout now owns the text-only work; the specification
and original framework checkouts remain unchanged. The tokenizer branch is
committed and pushed as
[PR #35](https://github.com/swiss-ai/apertus-omni-tokenizer/pull/35) against
`main`; framework changes remain local and unmerged. The tokenizer README and
framework `APERTUS2_INTEGRATION.md` files record commands and limitations;
the tested dependency overlay is retained alongside them as `framework-deps`.

Initial verification on October 6, 2026 (tokenizer `d00a5c1`; library with the
decisions recorded below): `make all` passes 323 tests, strict mypy and
pyright, 100% exported-type completeness, 17 rule invariants and execution/type checking of all four
notebooks. `make check-artifact` and `make harness` pass with the PR #35
instruct artifact. The `0.2.0` wheel and source distribution build; the
installed wheel passes the same 323 tests and model/parser imports leave
`tokenizers` and `jsonschema` unloaded.

A follow-up repository review on October 6 fixed empty call names reaching
streaming events before rejection: both parsers now fail at the header delimiter
with `INVALID_HEADER`. Text/token regressions bring `make all` to 325 passing
tests, with the same typing, API and notebook gates. The review also strengthened
expected-error assertions in the examples and reconciled the design docs with
BOS, checkpoint and mapping behavior. All five README/design Python examples
execute with the real instruct artifact. Real-artifact and harness checks pass;
wheel and source builds pass. External framework suites were not rerun in this
follow-up review.

Framework CPU suites against these revisions, after updating the adapters for
BOS, claim omission, the sidecar-free artifact and the vLLM fixes below: eight
Transformers and thirteen vLLM tests pass. These suites do not satisfy the engine/HTTP gates. The Transformers
round-trip case exercises two actual generations through a tool-call/result
round trip using the real tokenizer and a tiny random model with controlled
logits. It does not verify a trained Apertus checkpoint.

Local engine probing reports macOS ARM, vLLM `UnspecifiedPlatform`, no compiled
`vllm._C`, and an engine import blocked by missing `llguidance`. Full vLLM
engine/HTTP verification requires a configured supported execution environment
and model checkpoint; the CPU adapter suite does not substitute for that test.

On this machine, the parser benchmark measured approximately 4.9-35.5 million
characters/s for a long text payload and 0.76-2.88 million tokens/s for its token
representation across batch/small-chunk cases. Many-message cases measured
2.9-4.3 million characters/s and 0.49-0.60 million tokens/s. Traced Python peak
memory ranged from 1.7 to 34.8 MB. These are local observations from `make bench`,
not performance guarantees or measurements of native tokenizer allocations.

### External dependencies and release gates

- **Apertus 2 tokenizer rollout ([#1139](https://github.com/swiss-ai/apertus-program/issues/1139)):**
  PR #35 is open with passing CI and requested reviews from robmsmt and
  cimeister. Before merging, settle the special-token layout (PII placement,
  see decisions below). Then release the artifacts and verify the library
  against the published files. No tokenizer files are vendored into this
  library.
- **Message-boundary suspension:** the format says the engine stops only at
  `<|wait|>`, while the planned stateless vLLM implementation suspends requests
  at `<|/out|>`. The spec author's
  [#765 position](https://github.com/swiss-ai/apertus-program/issues/765#issuecomment-5233355020)
  calls `<|/out|>` in the engine stop list a broken configuration: the harness
  streams, and cancels and resubmits at a message boundary only when input is
  pending. Adopt that model or obtain explicit approval for stop-and-resubmit
  before declaring the integration conformant. Do not modify the read-only spec.
- **Framework rollout and release:** benchmarks, wheel/source builds, installed
  package checks and lightweight import checks pass as recorded above. Resolve
  the framework findings below, pin tested library and tokenizer revisions, and
  verify actual Apertus model generation, closing-token retention, mask
  enforcement and native continuation through vLLM's engine/HTTP path before
  declaring serving conformance. Publish only after the remaining rollout gates
  pass; unavailable checks remain pending.

### Open findings from the October 6 review

A report-only review compared the library with the profile at `85874b8`, the
tokenizer at `d00a5c1` and both framework checkouts. Upstream `v1` (`591701c`)
has no profile changes since `85874b8`.

**Library and profile alignment:**

- No loss-mask/segment helper for post-training data.

**Framework adapters:**

- Transformers: rebase onto the current fork head and re-verify.
- vLLM: core-file edits remain (registry entries, the parser-manager branch,
  the always-adjust hook, the startup flag check and the batch request reuse);
  propose generic hooks when upstreaming. Unverified performance: requests
  carry about 200,000 allowed token IDs, and the backend uses vLLM's slow
  incremental detokenizer.
- Both: replace the uncommitted library state recorded in
  `APERTUS2_INTEGRATION.md` with a pinned library commit once committed.

**Decisions after the review (October 6):**

- Resolved library findings: empty document `from`/`id` and empty tool,
  `call` and `result` names are rejected structurally (`native-v1.schema.json`
  adds `minLength: 1`; checker evidence rejects empty names and document IDs), a wait directly
  after a wait without new input is a `conversation/wait` violation, the
  empty-assistant mapping reports a diagnostic, the unused `HeaderError` is
  removed, and literal fixtures cover `result`, `think`, `claim` and unknown
  outputs with headers.
- Resolved vLLM findings: library errors become `ValueError` (HTTP 400), null
  tool fields and `strict: false` are accepted, the batch endpoint builds
  sampling parameters from the adjusted requests, startup requires
  `--tokenizer-mode apertus2` with `--tool-call-parser apertus2` and
  `--enable-auto-tool-choice`, and output cut off inside a message returns the
  completed messages with a warning while malformed framing still fails.

- Serving does not consume claims: both adapters map output with
  `MappingPolicy(omit_claims=True)`, the profile's inert behaviour. Transformers
  reports `claim-omitted` as a `UserWarning`; vLLM logs it at debug level.
  Exposing claims through an OpenAI extension field for RL or grading is
  recorded in the local `future.md`.

- The system JSON always contains `tools`, an empty array when no tools are
  declared, matching the profile example. Implemented with a literal fixture;
  the profile wording is proposed in
  [spec PR #11](https://github.com/swiss-ai/apertus-2-format-spec-proposal/pull/11).
- The library targets post-training only. The checker's `pretraining` purpose
  is removed: documents always require provenance and complete histories a
  system prompt. Pretraining-only gaps (raw-text scoring framing, document
  BOS/EOS wrapping) are out of scope.
- `encode_conversation` starts with the tokenizer's BOS (`<s>`, ID 1 in the
  instruct artifact); no EOS is added. Spec PR #11 adds this to the profile in a
  separate commit and lets a profile keep a leading BOS in the format text.
- The instruct `generation_config.json` keeps `</s>` as a second stop for now.
  Library constraints still use `<|wait|>` as EOS and forbid `</s>`, so masked
  generation cannot emit it; unmasked HF `generate` can stop on it.
- PII placement: keeping `<iban-pii>`, `<email-pii>` and `<ip-pii>` at base
  IDs 24-26 with the controls at 7-12 (`<|out|>`/`<|/out|>` at the base
  assistant IDs 10/11) is feasible. A prototype build passes `make
  check-artifact` and `make harness` with unchanged control and golden IDs;
  reserved slots become 4-6, 13-23 and the contiguous 27-123. Adopting it
  changes the tokenizer recipe and needs the PR #35 reviewers' agreement.

**Upstream questions:** the counter after an invalid counter ("next unused").
Spec PR #11 also removes padding spaces from profile example payloads and points
the one-token claims at the PR #35 artifact.

## Module and API conventions

Three nested scopes, one rule each. The pattern follows PEP 8, the
[typing spec's re-export rules](https://typing.python.org/en/latest/spec/distributing.html#import-conventions),
and the layout of httpx / pydantic v2. Single underscore only, at every
level; no `__` name-mangling.

### 1. Library level: the public API

The public API is exactly what the `__init__.py` files list in
`__all__`; nothing else is a contract.

- **Re-export is recursive:** every public name of a subpackage is also
  re-exported by its parent, so the flat top level
  (`from apertus_common import User`) always offers the whole API.
  Concretely: `__all__` of `apertus_common.model` is a subset of
  `__all__` of `apertus_common`.
- Re-exports are explicit named imports, never `import *`; presence in
  `__all__` (sorted) is what marks a name as intentionally public, for
  readers and for pyright/mypy alike. `tests/test_public_api.py` guards
  the surface.
- `__init__.py` files contain only docstring, re-exports, and `__all__`.
- One `py.typed` at the package root (`src/apertus_common/`); PEP 561
  marks the whole distribution as typed, subpackages never carry a copy.

### 2. Package level: filenames

Implementation modules are underscore-prefixed (`model/_base.py`).
Users import from the package, never from a `_*` path, so files can be
split, merged, or renamed without a breaking change.

### 3. Module level: names inside a `_*` file

The module underscore already makes everything in the file non-public;
the inner underscore is a second, finer signal:

- **Unprefixed** (`User`, `freeze_list`, `is_registered`): the module's
  export surface, meaning re-exported publicly via `__init__.py` or
  shared with sibling modules. Staying out of `__all__` is what keeps
  shared-only helpers package-private. Another package of this library
  may import an unprefixed name directly from the `_*` module when no
  public surface fits (the parser's reuse of the header grammar from
  `model/_base.py` is the precedent); external code never may.
- **Underscored** (`_coerce_payload`, `_REGISTRY`, `_MessageBase`):
  local to this module; siblings must not import it.

Rule of thumb: a sibling needs it, drop the underscore and keep it out
of `__all__`; only this module needs it, keep the underscore.

### Adding a public symbol

Define it unprefixed in its `_*` module, add it to the `__init__.py`
and `__all__` of its package, mirror it in every parent `__all__` up to
the top level, and update `tests/test_public_api.py`.
