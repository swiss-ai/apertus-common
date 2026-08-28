# apertus-common architecture

This document owns implementation architecture and the rationale behind major
design choices. [SPEC.md](../SPEC.md) owns library scope and public guarantees;
[roadmap.md](roadmap.md) owns sequencing, status, and unresolved work. External
ground truth for the token format is `spec.md` on the `v1` branch of
[swiss-ai/apertus-2-format-spec-proposal](https://github.com/swiss-ai/apertus-2-format-spec-proposal).

## 1. Architecture

### 1.1 Representations and data flow

The implemented codec has public **Model** and **String** representations plus
an internal **wire IR**. The planned tokenizer binding will add the **Tokens**
representation as a second projection of the same wire IR. Native JSON and
OpenAI messages are model adapters rather than wire codecs.

```
external forms (native JSON, OpenAI messages)
                  │ formats/
                  ▼
            Conversation ── render ──▶ Wire IR ── project ──▶ String

generated String ── TextLexer ──▶ Lexemes ──▶ OutputMachine ──▶ OutputMessage

planned: Wire IR ── tokenizer binding ──▶ Token IDs
planned: Token IDs ── TokenLexer ──▶ shared OutputMachine
```

- A **wire atom** is one atom of the wire form: `Control(token)` (framing),
  `Text(str, trust)` (ordinary text with its trust origin), or `Media(part)`
  (an inline placeholder). Framing exists in exactly one place: the
  Model-to-wire renderer. All separator whitespace is emitted as explicit
  structural `Text` atoms, so every projection is pure concatenation.
- String output is the implemented projection: control glyphs plus verbatim
  text. The planned token projection will map controls to registered IDs and
  encode ordinary text with special-token recognition disabled. Its required
  invariant is `encode(m) == tokenize(render_str(m))` on the unambiguous string
  subset (D2).
- The renderer preserves adjacent `Text` atoms, including separate adjacent
  `TextPart` values. The string projection joins them directly. The planned
  token projection must buffer each maximal run of `Text` and encode the
  concatenated run once, because source `TextPart` boundaries have no wire
  meaning and encoding the parts separately can change token IDs.
- The `Text.trust` tag records provenance only: `structural` text is written by
  the renderer and `payload` text comes from model content. It is not a
  tokenizer-call boundary because structural and payload atoms are commonly
  adjacent. In the planned token projection, all ordinary `Text` uses the
  specials-disabled encoder; registered IDs enter only through `Control` or
  typed media processing (D4).
- The implemented reverse direction is the experimental streaming text parser.
  `TextLexer` lifts generated text into shared lexemes, and one strict output
  machine exposes provisional deltas and completed typed messages. A token
  lexer, reference tool/reasoning parsers, and engine adapters are planned
  consumers of this boundary (D5). Failures stop the parser; consumers own
  recovery. Full transcript lifting is optional, and the checker stays
  model-only.

### 1.2 Layering

Dependencies point strictly downward; no cycles, no upward imports:

```
formats/     -> checker/, model/, errors.py
parsing/     -> core/, model/, errors.py
checker/     -> model/
core/        -> model/
model/       -> pydantic
```

`core/` contains the wire IR, renderer, text lexer, and output state machine.
Its incremental loops use plain slotted dataclasses and tuples; Pydantic runs
when constructing public models, reading formats, and closing a parsed output
message (D1). The planned tokenizer binding must preserve these dependency
directions.

### 1.3 Package layout

```
src/apertus_common/
  model/        # messages, headers, system prompt, conversation
  checker/      # conversation-level conformance rules and check()
  core/         # wire IR, renderer, text lexer, output machine
  parsing/      # experimental public streaming text parser
  formats/      # native JSON and OpenAI request mapping
tests/
  test_*.py     # model, checker, formats, renderer, and parser tests

planned:
  tokenizer/          # artifact binding and token-ID frontend
  tests/golden/       # exact token IDs and parser results
  tests/equivalence/  # optional Jinja diagnostics
```

### 1.4 Public and planned top-level API

The package namespace is the API; internals stay private (see
[CONVENTIONS.md](../CONVENTIONS.md)). The representative block below labels
planned entries explicitly:

```python
from apertus_common import (
    # model
    Conversation, SystemPrompt,  # + the concrete message cast
    # implemented codec
    render_str,              # Conversation -> string form
    # checker
    check,                   # Conversation -> ConformanceReport
    # streaming
    StreamParser,            # experimental strict output-only text parsing
    # implemented formats
    from_json, to_json,      # the model's own JSON under harmony-style names,
                             # ingest flags normalize / check / strict
    from_openai, to_openai,  # OpenAI chat-completions mapping, lossy, stated subset
    ConformanceError,        # raised by strict=True (carries the report)
)

# Planned after tokenizer integration:
# from apertus_common import encode
```

Optional extensions are scoped in [SPEC.md §5](../SPEC.md#5-optional-later-extensions).
Parser-specific variants are detailed in [parsing.md](parsing.md).

The formats boundary is complete. Native JSON uses `from_json` and `to_json`;
OpenAI chat-completions conversion uses `from_openai` and `to_openai`; all four
are exported at the package root. Format lifts construct the typed model before
the core renderer or checker runs, so external representations never author
wire framing directly. Dataset files use the JSON-lines convention from D18.

## 2. Design decisions

**D1: Pydantic at the boundary, plain objects in the core.** The typed model
is Pydantic v2: validation, schema export, serialization for free. The core
never runs Pydantic validation per token or wire atom; Model-to-wire
converts once, then everything is plain data.

**D2: The wire IR is the single source of framing truth.** Only the
renderer produces framing (mechanics in 1.1); any spec change to the wire
form lands in one file.

**D3: The string form is a library-defined codec, not a template.**
Control tokens appear as their surface glyphs; payload text renders
verbatim (O1 interim: no escaping, no content validation). A payload that
itself contains a control glyph therefore renders ambiguously in the
string projection. The planned token path avoids this ambiguity by encoding
payloads with special-token recognition disabled (D4) and is intended to be
the fully faithful wire serialization.
Accordingly, string parsing (the streaming output parser now, the
optional `parse_str` later) is a partial inverse until escaping exists:
its contract excludes ordinary content containing control glyphs or
ambiguous system-tag delimiters. Such input may parse as different
structure and must not be treated as an untrusted transport format.
Deferred, string path only: optional content validation by exact string
match against the artifact's special-token inventory, and, if
losslessness is later required, the escaping rule (in payloads,
`<(\*)\|` gains one backslash on render and loses one on parse).
The planned String-to-Tokens path must **not** tokenize with specials enabled.
For input in the unambiguous subset, it will lift the string back to wire atoms
and encode ordinary text specials-disabled. Untrusted or ambiguous strings are
outside this path's contract; the typed-model and future token-ID paths retain
the format's full unforgeability guarantee until escaping exists.

**D4: Planned token projection uses two disjoint paths.** Every special ID will
be placed by code walking validated typed structure; every byte of text will go
through one encoder that cannot produce special IDs. Concretely: framing
IDs will come from an explicit `ControlTokens` name-to-ID map and be appended
directly, never tokenized; every ordinary `Text` run will be encoded with
special-token recognition disabled (the tokenizer's
`encode_special_tokens` switch), so a payload spelling a control glyph yields
inert ordinary tokens; media placeholder and expansion IDs will be emitted from
typed parts and processor output, never from text. The tokenizer binding will
expose exactly one text encoder, hardwired to specials-disabled; no API will
encode ordinary text any other way. The typed part list is the trust boundary
for media.

Coalescing will be a token-projection operation, not an IR normalization step.
Encoding adjacent pieces separately can change BPE token ids, while physically
merging wire atoms would discard useful provenance and future source mapping.
When optional token metadata such as loss masks lands, the internal atoms may
gain compact message ownership and semantic-region labels. Recipe-specific
mask values remain derived output and must not affect coalescing or token ids.

**D5: One strict output parser core, streaming-first.** The implemented text
frontend produces source-neutral lexemes for one output state machine. The
planned token frontend will target the same boundary. Batch parsing is the
completed form of the text path. The machine reports the first failure and
becomes terminal; consumers alone choose recovery. This keeps parsing separate
from model-only conformance checking and leaves full transcript lifting as a
separate optional consumer. The experimental interface, grammar, state, queue,
lifecycle, and failure contracts are defined in [parsing.md](parsing.md).

**D6: One fully supported form, explicit conversions for the rest
(implemented).** Native model JSON is the canonical serialization. `formats/`
provides plain
`from_json`/`to_json` and `from_openai`/`to_openai` functions, with no protocol,
registry, or file helpers. Shared ingest flags compose normalization and
conformance reporting as described in D9. Unsupported outbound mappings raise
instead of being dropped or relabelled. Dataset framing remains a convention
(D18). The planned OpenAI response side belongs in a serving adapter over
`StreamParser`; it is not implemented by `to_openai`. Exact public adapter
guarantees are owned by SPEC.md §3.4 and the function documentation.
The complete public interface and conversion tables are in
[formats.md](formats.md).

**D7: Structured errors.** Each boundary reports stable typed failures with
locations and chained causes where applicable. Parsing owns its exception
hierarchy and caller-misuse error, defined in [parsing.md](parsing.md). Formats
use focused adapter errors, strict ingest carries a conformance report, and
the checker returns typed violations instead of a bare boolean. Consumers do
not need to match human-readable error strings.

**D8: Optional Jinja equivalence remains outside the runtime.** A later
test-only suite may render the same conversations via the library and a Jinja
template, compare token-for-token on the representable subset, and keep an
explicit exclusion list. The tokenizer may also ship a raising stub directing
users to this library. Neither is part of the core completion contract. *Open:
O2.*

**D9: Two read modes: normalize on ingest, report on conformance.** Every
cross-message rule has one definition (`TypeInfo` in the model layer) and
two consumers with opposite behavior. The **ingest path** (authoring,
lifting external formats) corrects: `Conversation.normalized()` sorts input
runs into canonical order, and `formats` lift normalizes by default. The
**conformance path** (`check()` over a directly constructed or successfully
parsed `Conversation`) reports: the checker emits an ordering violation and
never silently reorders. A read-and-fix pipeline is the composition:
obtain a `Conversation` (constructed, format-lifted, or via the optional
transcript parser), `check()` it, then `normalized()` when the caller
explicitly accepts correction; the ingest flags of `from_json`/`from_openai`
(`normalize`, `check`, `strict`) are that composition in one call (D6).

Decisions specific to the typed model (implemented):

**D10: Labels as metadata.** Delivery order and authority rank are
`ClassVar`s exposed via the `TypeInfo` registry, never serialized (spec §5:
not fields in the message); `direction` and `type` *are* fields carried by the
wire form. Unranked input types default to order 3, authority 4.
Rank is semantic metadata, not an encoding switch. The planned token projection
will encode all payload text with special-token recognition disabled (D4).

**D11: Payloads.** `User`/`Attachment` take part lists
(`TextPart | ImagePart | AudioPart`; a bare `str` coerces); media parts hold
opaque handles, expansion is the processor's job. `ToolCall` is typed fields
(`id`, `name`, `args`: any finite JSON value); the wire JSON of spec §6 is
produced by the codec layer. Everything else is `str`.

**D12: Headers.** Typed known keys plus an open `extras` mapping;
conservative value grammar (including a `<|`/`|>` glyph guard);
non-conforming values are rejected, never escaped, until O6 settles.

**D13: Ids.** Permissive `ToolCallId` (`[A-Za-z0-9_:.-]{1,64}`) plus
`is_canonical_id()` for the `NAME:COUNTER` convention, pending O5.

**D14: Local vs. global validation.** The model validates fields and grammar
eagerly; cross-message rules such as canonical order, id echo, and wait
placement belong to the checker, so non-conformant conversations stay
representable (D5, D9). Memory policy is outside this library.

**D15: Normalization lives on the model.** `Conversation.normalized()` is
the correcting half of D9: each maximal run of consecutive input messages is
stably sorted by `TypeInfo` order (unlabeled types key at position 3);
outputs and waits never move; idempotent; never applied implicitly by
validation or parsing.

**D16: Conversation-level checkers, one class per spec rule family.** The
conformance checker is a set of rule-family classes over the typed
`Conversation` in `checker/rules/`, one module per family, each module
docstring stating the rule it implements in its own words (code stays
self-contained and independent of spec numbering; the spec mapping lives
in this document). `check()` runs a strict
default set and returns a `ConformanceReport` (D7); it reports, never
raises (D9/D14). Extension is subclassing `Checker` and passing instances;
`checkers=` replaces the set (extend with `[*default_checkers(), mine]`);
no registry, no entry points. Every rule is declared as data (`Rule`: id,
description, and default severity), so `rule_table()` yields the
machine-readable rule inventory. **Rule ids are the stable contract;
checker class granularity may be reorganized.** The `family/` prefix is
the id namespace: builtin families are reserved, third parties pick a
distinct family, and `check()` rejects an id declared by two checker
types. Severities are error (breaking) or warning (non-breaking): strict
by default, loosenable per checker (`severity=`), per rule (constructor
`overrides=`), or per call (`check(..., overrides=)`). Warning defaults
identify advisory or unsettled rules: the id convention
(O5), open calls at the end of a transcript, verifiable-answer placement
and cardinality, and an unterminated trailing output run. `jsonschema` is
a runtime dependency for args-vs-schema checking; the declared schema
dialect is unspecified upstream, so an explicit `$schema` is respected
and the default is Draft 2020-12 (candidate to raise upstream); schema
evaluation never retrieves remote references. Division of labor with
parsing: framing and payload syntax live in the parsing layer (the streaming
parser stops and raises a structured exception); `check` owns semantic and cross-message conformance over
a complete typed model. The public checker does not accept raw strings
or partial parse results.

**D17: Canonical string rendering conventions.** The spec declares glyphs
and spacing non-normative, so the library's conventions are the reference
rendering, fixed from the spec's worked examples: message form
`<|in|> HEADER <|hdr|> PAYLOAD <|/in|>` (same with out) with exactly one
space between elements, an empty payload collapsing to a single space;
headers as space-separated `key=value` with `type` first, then the typed
keys in field order, then extras in insertion order; messages separated by
one blank line; `<|wait|>` on its own line attached to the preceding block
(a conversation-initial wait stands alone; placement conformance is the
checker's concern); no leading or trailing newline. The system prompt renders compact: one tag
per line, absent optional tags omitted, tag order identity, behavior,
effort, tools, extra tags, environment last; `<tool>` entries nested at
2/4-space indent with compact JSON schemas. Tool call payloads render as
compact JSON `{"name":...,"args":...}` with unicode preserved. Payload
part lists concatenate with no injected separators; media parts render as
their placeholder glyphs, expansion stays with the processor. Padding and
generation prefixes are later concerns.

**D18: Datasets are JSON lines of the typed model, by convention
(implemented).** One `Conversation` JSON per line (`to_json`, i.e.
`model_dump_json()`;
`system` null for pretraining data), so a corpus is lossless by
construction, its line schema is `Conversation.model_json_schema()`
(external tools validate without the library), and it is versioned by the
library release that owns that schema. No per-line envelope, version key,
or dataset metadata (a pipeline wraps and unwraps its own), and no library
helper: reading is `from_json` per line, with the ingest flags; writing is
`to_json` plus a newline. Lines written this way round-trip byte for byte
through `from_json(normalize=False)`, and through the default when their
input order is already canonical. Authoring conveniences are not part of
the convention (closes O4).
