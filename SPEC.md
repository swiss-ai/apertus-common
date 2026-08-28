# apertus-common Library Specification

Status: **authoritative library-scope contract**. The Apertus 2 format spec
governs wire semantics; this document defines what this repository provides.
Implementation architecture lives in [docs/design.md](docs/design.md).
Sequencing, status, and unresolved work live in
[docs/roadmap.md](docs/roadmap.md).

## Table of contents

- [1. Purpose](#1-purpose)
- [2. Normative references](#2-normative-references)
- [3. Core scope](#3-core-scope)
  - [3.1 Typed conversation model](#31-typed-conversation-model)
  - [3.2 Tokenizer binding and codecs](#32-tokenizer-binding-and-codecs)
  - [3.3 Parser and conformance checker](#33-parser-and-conformance-checker)
  - [3.4 OpenAI API adapter](#34-openai-api-adapter)
  - [3.5 Golden conformance contract](#35-golden-conformance-contract)
- [4. Core serving rules](#4-core-serving-rules)
- [5. Optional later extensions](#5-optional-later-extensions)
  - [5.1 Loss-mask emission](#51-loss-mask-emission)
  - [5.2 Jinja stub and equivalence testing](#52-jinja-stub-and-equivalence-testing)
  - [5.3 Engine plugins](#53-engine-plugins)
  - [5.4 Raw-text and pretraining helpers](#54-raw-text-and-pretraining-helpers)
- [6. Design constraints](#6-design-constraints)
- [7. Non-goals](#7-non-goals)
- [8. Project planning](#8-project-planning)

## 1. Purpose

`apertus-common` is the typed reference implementation of the Apertus 2
conversation format. It is intended to give training, serving, and evaluation
one path for constructing token sequences and parsing them back into messages,
preventing format drift between independently maintained templates and parser
flags. The current package provides the typed model, checker, string renderer,
format adapters, and an experimental streaming text parser; tokenizer and
serving integration remain planned.

## 2. Normative references

- **Format semantics:** `spec.md` on branch `v1` of
  [swiss-ai/apertus-2-format-spec-proposal](https://github.com/swiss-ai/apertus-2-format-spec-proposal)
  (local: `/Users/rkre/swissai_repos/apertus-2-format-spec-proposal`). It owns
  framing, control tokens, message types, order and authority, generation and
  halt states, multimodal payloads, system prompts, and pretraining framing.
- **Tokenizer artifact:** built by
  [swiss-ai/apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer)
  (local branch `v2-on-pr24`) as `tokenizers/Apertus_2` via
  `python -m omnitok.cli build-apertus-2`. The planned tokenizer binding will
  consume its vocabulary, control-token IDs, and media metadata; this library
  never vendors or hand-edits them.
- **Program decisions:**
  [#856](https://github.com/swiss-ai/apertus-program/issues/856) tracks the
  library and conformance pilot;
  [#765](https://github.com/swiss-ai/apertus-program/issues/765) records the
  format and serving review.

The format spec wins on format questions; this document wins on library scope
and API. Conflicts must be raised upstream, not silently resolved in code.

## 3. Core scope

### 3.1 Typed conversation model

The public model represents the complete format cast:

- system prompt, tools, wait, headers, and multimodal text/image/audio parts;
- inputs: `harness`, `tool_result`, `retrieval`, `event`, `attachment`, `user`,
  and pretraining `document`;
- outputs: `assistant`, `think`, `tool_call`, and `verifiable_answer`;
- deployment-defined message types and header keys through generic models.

Models validate eagerly, serialize losslessly, export JSON Schema, and preserve
exact message classes on round-trip. Order and authority are typed metadata,
not wire fields. Cross-message violations remain representable for the checker.

### 3.2 Tokenizer binding and codecs

`render_str` is implemented. The tokenizer binding, `encode`, and token-ID
frontend are required by this contract but are not implemented yet.

The core conversion path is:

```text
Conversation ⇄ Wire IR (Control | Text | Media) ⇄ String / Token IDs
```

- The wire IR (atom sequence) is the single source of framing truth.
- `render_str` provides the string representation with verbatim payloads.
  Until escaping is defined, parsing and round-trip guarantees cover only
  the unambiguous subset: ordinary content containing control glyphs or
  ambiguous system delimiters is outside the string-codec contract.
- `encode` will bind wire atoms to the tokenizer artifact; full transcript
  `parse_str`/`decode` are optional later extensions.
- Untrusted payload text is always tokenized with special-token recognition
  disabled; no payload API may bypass this rule.
- Canonical input order is applied explicitly through normalization before
  encoding; the encoder never silently changes observed order. It renders the
  typed system prompt in canonical tag order, including required `<effort>`.

### 3.3 Parser and conformance checker

One streaming-first, output-only parser consumes a contiguous model-output
segment (text chunks now, token IDs once the artifact converges) through a
source-neutral lexeme interface. It tracks message open, header, payload,
close, separator, and wait transitions as internal state. Its public face is
`StreamParser`: `push` returns nothing, while consumers drain ordered
provisional payload deltas and completed typed messages. Planned reference tool
and reasoning parsers will consume the same state.

`StreamParser` is strict and stops at the first failure. It raises a structured
exception carrying the failure reason, source position, raw uncommitted
fragment, unconsumed suffix, and the exact terminal parser instance. This
keeps retained messages and queued deltas inspectable even when the batch
entry point created the parser internally. It never skips to a later boundary
or selects a fallback. Consumers decide whether to retain completed messages,
use provisional deltas, expose raw output, or start a new parser on a chosen
suffix. The conformance checker remains separate and accepts only a
successfully constructed `Conversation`.

Malformed framing and incomplete input never enter the completed-message
queue. Consumers that require the same guarantee for visible output buffer
provisional deltas until message completion. Header grammar and duplicate keys
are enforced. The optional later `parse_str` is strict and stops at the first
failure:
it either parses and validates one
complete `Conversation` or raises an exception distinguishing framing,
incomplete input, and model validation. `check` continues to accept only a
complete typed `Conversation` and returns its full semantic and cross-message
report; parsing failures are not converted into conformance findings.

### 3.4 OpenAI API adapter

The adapter contract follows the decisions from #765:

- `system` maps to the system prompt; a later `system` message maps to
  `harness`; `user`, `assistant`, and `tool` map to their typed equivalents.
- `think` maps only to `reasoning_content`, in arrival order. Multiple
  reasoning/content switches are valid; reasoning bytes never enter `content`.
- `finish_reason` is `stop` on wait without open calls, `tool_calls` with open
  calls, and `length` on token budget.
- Assistant prefill opens an assistant output message for model continuation.
- Tool-call IDs matching `[A-Za-z0-9_:.-]{1,64}` round-trip byte-for-byte;
  model-written IDs conventionally use `NAME:COUNTER`.

The request-side `from_openai` and `to_openai` mappings are implemented.
Response conversion, including `finish_reason`, remains planned and will be
owned by a serving adapter over `StreamParser` state.

The complete public conversion interface, mapping tables, loss boundaries,
and error behavior are documented in [docs/formats.md](docs/formats.md).

### 3.5 Golden conformance contract

Golden conformance files are required by this contract but are not implemented
yet.

Golden fixtures will map typed conversations to exact token IDs, rendered
strings, and parser results. Round-trip and malformed-input fixtures will cover
every serving rule. External consumers will be able to run these files without
importing the library; the deterministic fixtures, not prose alone, will be the
compatibility contract.

## 4. Core serving rules

- `<|wait|>` is the only stop token. `<|/out|>` is an interruption opportunity,
  never an engine stop; think-then-answer fixtures must return the answer.
- Engines suppress input-framing tokens during decoding.
- Untrusted payload text is encoded with special-token recognition disabled.
- At each boundary, inputs follow canonical order: harness, tool result,
  retrieval/event, attachment, user.
- A `tool_call` payload is exactly `{"name": NAME, "args": ARGS}` and `ARGS`
  conforms to the declared tool schema.
- Unknown and known-but-unconsumed output types are distinct; neither interrupts
  decoding. Unknown types may trigger a recovery notice from the harness.
- Types without an explicit deployment rank default to authority rank 4.

## 5. Optional later extensions

Full transcript lift: strict `parse_str` (string to one validated
`Conversation` or a distinguishable exception), token `decode`, and
`check` over strings or token ids. Buildable on the same parser core;
not part of the current surface or milestones.

These features build on the core contract but are not required to complete the
core model, codec, parser, adapter, and conformance path.

### 5.1 Loss-mask emission

The renderer may emit training loss masks alongside token IDs. Input framing is
never a target; output framing is a target when its message carries loss; think
and annotation loss remains a training-recipe choice.

### 5.2 Jinja stub and equivalence testing

The tokenizer may ship a chat-template stub that raises and directs callers to
this library. A test-only Jinja implementation may be compared token-for-token
on the subset it can represent, with explicit exclusions for unforgeability,
multi-message generation, splicing, and memory policy.

### 5.3 Engine plugins

vLLM and SGLang tool/reasoning parser plugin classes may wrap the core parser so
deployments can integrate before equivalent support lands upstream. The core
parser and golden fixtures remain engine-independent.

### 5.4 Raw-text and pretraining helpers

Convenience helpers may wrap raw text as `type=document`, split documents into
complete messages, pack sequences with `<|pad|>`, and keep document plus safety
annotation `think` pairs together. Bare unframed text remains out of distribution.

## 6. Design constraints

- **One framing implementation:** no parallel template or adapter may author
  control-token sequences independently.
- **Typed boundaries, plain hot path:** Pydantic validates public data; core
  wire-atom/token loops use lightweight Python values.
- **Streaming-first reverse path:** any non-streaming parsing (an optional
  extension) is the completed form of the same incremental state machine.
- **Structured failures:** errors and conformance reports carry type, rule, and
  location rather than a bare boolean.
- **Pure Python first:** optimize or port only after profiling.

## 7. Non-goals

- No tool execution, scheduling, conversation splicing, or memory-policy
  enforcement: those belong to a harness or agent runtime.
- No inference engine implementation.
- No second working production codec hidden in a Jinja template.
- No tokenizer files maintained or patched in this repository.
- The conformance pilot model is a consumer of the library, not library code.

## 8. Project planning

Milestones, implementation status, and open decisions are maintained in
[docs/roadmap.md](docs/roadmap.md). Contract changes must still be reflected in
this document once settled.
