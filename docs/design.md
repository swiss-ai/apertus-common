# Implemented design

This document describes the code and public behavior currently implemented in
`apertus-common`. Development scope, conventions, future work, and unresolved
questions live in [DEVELOPMENT.md](../DEVELOPMENT.md). The
[README](../README.md) provides an overview and the next tasks.

The implementation is a text-only pilot of the earlier conversation format.
It has not yet been adapted to the current Apertus 2 profile. The experimental
streaming parser is implemented, but its interface is not considered stable.

## Architecture

```text
Native JSON / OpenAI messages
              | formats/
              v
        Conversation -> wire IR -> rendered string

generated string -> TextLexer -> lexemes -> OutputMachine -> output messages
```

| Package | Responsibility |
|---|---|
| `model/` | Immutable typed messages, system prompts, conversations, normalization |
| `checker/` | Conversation-level conformance reports and rule metadata |
| `core/` | Wire atoms, renderer, text lexer, output state machine |
| `parsing/` | Public `StreamParser`, message construction, queues, lifecycle |
| `formats/` | Native JSON and OpenAI message adapters |

Dependencies point downward: `formats` uses `checker`, `model`, and errors;
`parsing` uses `core`, `model`, and errors; `checker` and `core` use `model`;
`model` uses Pydantic. Core loops use plain dataclasses and Python values.
Pydantic validates public data at construction and once per completed parsed
message, not per lexeme.

Public names are explicitly exported in package `__all__` lists and recursively
re-exported up to `apertus_common`. Implementation modules are private.
[Development conventions](../DEVELOPMENT.md#module-and-api-conventions) describe
how to maintain those boundaries.

## Typed model

`Conversation` contains an optional `SystemPrompt` and ordered `items`:

- Inputs: `Harness`, `ToolResult`, `Retrieval`, `Event`, `Attachment`, `User`,
  `Document`, and `GenericInput`.
- Outputs: `Assistant`, `Think`, `ToolCall`, `VerifiableAnswer`, and
  `GenericOutput`.
- `Wait`, a separate marker for the model's deliberate stop.

Pydantic models and nested sequence/mapping containers are immutable.
Serialization restores exact message classes using `(direction, type)`
discrimination; unknown types select the generic model for their direction.
`Conversation.model_json_schema()` describes native conversation JSON.

`User` and `Attachment` carry immutable sequences of `TextPart`; a bare string
coerces to one text part. Text attachments remain supported. Other message
payloads are strings, except `ToolCall`, which has typed `id`, `name`, and
finite JSON `args` fields. Image and audio parts are not supported.

Headers have typed known fields and an open `extras` mapping. Keys and values
are validated eagerly; values cannot contain whitespace, `=`, `<|`, or `|>`.
Tool-call IDs accept `[A-Za-z0-9_:.-]{1,64}`;
`is_canonical_id()` checks the pilot's `NAME:COUNTER` convention.

Order and authority are metadata exposed through `type_info()`, never wire
fields. Unknown inputs use order 3 and authority 4. `Conversation.normalized()`
stably sorts each consecutive input run by delivery order; outputs and waits
never move. Direct model construction and rendering preserve observed order.

## Conformance checker

Model validation rejects structurally invalid values. Cross-message violations
remain representable, and `check(conversation)` reports them without modifying
the conversation or raising for violations.

Seven checker families cover system prompts, input order, tool calls, tool
results, IDs, waits, and verifiable answers. Rules have stable IDs,
descriptions, and error/warning severities. `rule_table()` exposes the inventory;
`default_checkers()` returns the default checker instances.

A `ConformanceReport` contains typed violations with locations, exposes `ok`,
`errors`, and `warnings`, and can be serialized with `dataclasses.asdict()`.
`checkers=` replaces the default checker set; subclass `Checker` to extend it.
Severity overrides are accepted per checker, per rule, and per call. Duplicate
rule IDs across checker types and unknown override IDs are rejected.

Tool argument validation uses `jsonschema`, respects an explicit `$schema`,
and defaults to Draft 2020-12. It never retrieves remote schema references.
Noncanonical IDs, open calls at transcript end, verifiable-answer placement
and cardinality, and an unterminated output run are advisory warnings.

## Wire IR and string rendering

Only the renderer authors framing. It walks the typed conversation into
`Control(token)` and `Text(text, trust)` atoms. Text provenance is `structural`
for renderer-authored syntax and `payload` for content. Separate adjacent text
parts remain separate atoms; string projection concatenates them directly.

`render_str()` emits control glyphs and verbatim text without escaping:

```text
<|in|> type=user <|hdr|> Hello <|/in|>

<|out|> type=assistant <|hdr|> Hi <|/out|>
<|wait|>
```

The implemented pilot inventory is `<|sys|>`, `<|/sys|>`, `<|in|>`, `<|/in|>`,
`<|out|>`, `<|/out|>`, `<|hdr|>`, `<|wait|>`, and `<|pad|>`. Padding is recognized
by the lexer but not emitted by the renderer or accepted by the output parser.
Former media glyphs such as `<|image|>` and `<|audio|>` are ordinary text with
no media semantics.

Headers render `type` first, then typed keys, then extras in insertion order.
Messages have one structural space between framing elements; an empty payload
uses one space between header terminator and closing token. Messages are
separated by a blank line. A wait occupies its own line attached to the prior
block. There is no leading or trailing newline.

The system prompt uses the pilot's `<|sys|>` envelope and XML-like tags in
order: identity, behavior, required effort, tools, extra tags, environment.
Absent optional tags are omitted. Tool schemas and call payloads use compact
JSON with Unicode preserved; a call payload is exactly `{"name":...,"args":...}`.

The string codec is ambiguous when ordinary content spells control glyphs or
system-tag delimiters. Its round-trip guarantees cover only unambiguous text;
rendered text is not an untrusted transport format. There is no token encoder
or full-transcript decoder in the current public API.

## Native JSON and OpenAI message formats

The `apertus_common.formats` boundary converts external data to and from the
typed `Conversation` model. It does not render Apertus wire framing. Import all
public functions from `apertus_common`:

```python
from apertus_common import (
    AdapterError,
    ConformanceError,
    from_json,
    from_openai,
    to_json,
    to_openai,
)
```

| Format | Read | Write | Contract |
|---|---|---|---|
| Native model JSON | `from_json` | `to_json` | Lossless with `normalize=False`; normalizes by default |
| OpenAI chat completions | `from_openai` | `to_openai` | Compatibility adapter with a stated lossy subset |

### Shared ingest controls

Both read functions construct and validate a `Conversation`, then apply the
same keyword controls:

```python
normalize: bool = True
check: bool = False
strict: bool = False
```

| Controls | Result |
|---|---|
| defaults | Normalize canonical input order and return `Conversation` |
| `normalize=False` | Preserve observed order |
| `check=True` | Return `(conversation, ConformanceReport)` |
| `strict=True` | Raise `ConformanceError` when the report has errors; warnings pass |
| `check=True, strict=True` | Raise on errors, otherwise return the conversation and report |

Normalization runs before checking. Use `normalize=False, check=True` to audit
the source as received. Local model validation always happens first and is not
converted into a conformance finding.

```python
conversation, report = from_json(text, normalize=False, check=True)

try:
    conversation = from_openai(messages, normalize=False, strict=True)
except ConformanceError as error:
    print(error.report.errors)
```

### Native JSON

```python
from_json(
    text: str | bytes,
    *,
    normalize: bool = True,
    check: bool = False,
    strict: bool = False,
) -> Conversation | tuple[Conversation, ConformanceReport]

to_json(conversation: Conversation) -> str
```

Native JSON is exactly the typed model serialization. It preserves concrete
message classes and is described by `Conversation.model_json_schema()`.

```python
from apertus_common import Conversation, User, from_json, to_json

source = Conversation(items=[User(payload="hello")])
text = to_json(source)
restored = from_json(text, normalize=False)
assert restored == source
```

`from_json` raises Pydantic `ValidationError` for malformed JSON or invalid
model fields. `to_json` performs no normalization or checking.

#### JSON-lines datasets

A dataset is one native conversation JSON value per line. There is no separate
file-format object or loader:

```python
with open("conversations.jsonl", encoding="utf-8") as source:
    conversations = (from_json(line) for line in source)

with open("out.jsonl", "w", encoding="utf-8") as sink:
    sink.writelines(to_json(conversation) + "\n" for conversation in conversations)
```

The caller owns file concerns such as line numbers, blank-line policy, skipping
bad records, and dataset metadata.

### OpenAI messages into the model

```python
from_openai(
    messages,
    tools=None,
    *,
    effort="medium",
    continue_final_message=False,
    normalize=True,
    check=False,
    strict=False,
) -> Conversation | tuple[Conversation, ConformanceReport]
```

Pass the request's `reasoning_effort` as `effort`. A `SystemPrompt` is always
created because Apertus requires effort.

| OpenAI input | Apertus model |
|---|---|
| Leading `system` or `developer` | `SystemPrompt.behavior` |
| Later `system` or `developer` | `Harness` |
| `user` text | `User` with `TextPart` |
| `user` `text` parts | `TextPart` |
| Assistant `reasoning_content` | `Think` |
| Assistant `content` | `Assistant` |
| Assistant `tool_calls` | `ToolCall`; ids are preserved and JSON arguments are parsed |
| `tool` | `ToolResult` using `tool_call_id` |
| Function tools | `ToolSpec`; parameters become the schema and description becomes policy |

Each assistant run is closed with `Wait`. Set
`continue_final_message=True` to leave only the trailing run open for assistant
prefill. An open run is representable and normally produces a checker warning.

Missing or empty tool arguments become `{}`; mapping arguments are accepted as
already parsed values. Missing tool parameters default to an object schema. An
assistant with no reasoning, content, or calls becomes an empty `Assistant`.
System, assistant, and tool content supplied as text-part lists is concatenated.

The adapter ignores OpenAI `name` fields and unknown keys. Only text content
parts are supported; other content parts raise `AdapterError`. Default
normalization can move a later system
notice, represented as `Harness`, ahead of user or tool inputs at the same
boundary. Use `normalize=False` when arrival order must be preserved.
Unsupported roles, malformed tool calls, and unsupported content-part shapes
raise `AdapterError`.

```python
request = {
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
}

conversation, report = from_openai(
    request["messages"],
    effort="high",
    check=True,
)
```

### Model into OpenAI messages

```python
to_openai(conversation: Conversation) -> dict[str, Any]
```

The result always contains `messages`. It also contains `tools` when declared
and `reasoning_effort` when the conversation has a system prompt.

| Apertus model | OpenAI output |
|---|---|
| `SystemPrompt` identity, behavior, environment | One `system` string joined with blank lines |
| `SystemPrompt.tools` | Function tools |
| `Harness` | `system` |
| `User` | `user`, as text or text parts |
| `ToolResult` | `tool` |
| `Think` | Assistant `reasoning_content` |
| `Assistant` | Assistant `content` |
| `ToolCall` | Assistant `tool_calls` with compact JSON arguments |
| `Wait` | Flush the current assistant message |

One OpenAI assistant message can contain one reasoning segment, one content
segment, then any number of tool calls. A repeated segment or a different
order starts another assistant message. This preserves reasoning separately
from content.

The following values have no OpenAI counterpart and raise `AdapterError`
instead of being dropped or relabelled:

- `Retrieval`, `Event`, `Attachment`, `Document`
- `GenericInput`, `GenericOutput`, `VerifiableAnswer`
- non-empty message header `extras`
- system prompt `extra_tags`

The adapter preserves the supported message content, but normalizes its
representation: singleton user text-part lists become strings; adjacent
assistant messages may merge; empty reasoning/content segments may be omitted; `developer` returns as `system`; assistant `content` is always
present and may be `None`; tool arguments are re-serialized compactly; system
identity, behavior, and environment lose their tag boundaries; ignored OpenAI
keys do not return.

```python
openai_request = to_openai(conversation)
messages = openai_request["messages"]
```

### Errors

| Operation | Error | Inspection |
|---|---|---|
| `from_json` | Pydantic `ValidationError` | Standard Pydantic locations and messages |
| Either read with `strict=True` | `ConformanceError` | `error.report` |
| `from_openai` mapping failure | `AdapterError` | `where="message"`, message `index`, `reason`, optional `__cause__` |
| Invalid OpenAI effort or tools | `AdapterError` | `where="system"`, `index=-1` |
| `to_openai` unsupported item | `AdapterError` | `where="item"`, item `index`, `reason` |
| `to_openai` unsupported system data | `AdapterError` | `where="system"`, `index=-1` |

Adapters stop at the first error. Skipping, repairing, or collecting bad
records is caller policy.

For executable dataset and conversion workflows, see
[the dataset notebook](../examples/dataset_examples.ipynb).

## Streaming output parser

`StreamParser` is an experimental, strict text parser for one contiguous
model-output segment. It returns provisional payload deltas and completed
validated output messages. It does not parse full transcripts or token IDs.
`check()` accepts only a constructed `Conversation`, not partial parser state.

### Public interface

```python
from apertus_common import StreamParseError, StreamParser

parser = StreamParser()
try:
    for chunk in engine_stream():       # arbitrary text chunk sizes
        parser.push(chunk)               # returns None
        for delta in parser.take_deltas():
            route_provisional(delta)
        for message in parser.take_completed():
            route_committed(message)
        parser.current_type              # open output type, or None
        parser.current_headers           # immutable parsed header mapping
        parser.current_payload           # accumulated open payload
        parser.waiting                   # wait token accepted
    parser.finish()
except StreamParseError as error:
    assert error.parser is parser
    handle_failure(error.parser, error)   # consumer policy

try:
    parsed = StreamParser.parse(text)     # fresh parser: push once, finish
except StreamParseError as error:
    inspect_partial(error.parser)         # terminal batch parser remains available
else:
    parsed.messages                       # cumulative completed messages
```

`PayloadDelta` is a frozen slotted dataclass:

```python
@dataclass(frozen=True, slots=True)
class PayloadDelta:
    ordinal: int
    type: str
    text: str
```

- `push(chunk) -> None` accepts any non-empty or empty `str` chunk. A chunk may
  cross payload, message, and wait boundaries.
- `take_deltas()` drains provisional `PayloadDelta` records in source order.
  Each record identifies its output message, so one push can safely cross
  several output types.
- `take_completed()` drains newly completed typed messages. `messages` remains
  cumulative for the parser's lifetime; draining does not remove history.
- `current_type`, `current_headers`, and `current_payload` describe only the
  currently open output message. They are `None`, an empty mapping, and `None`
  respectively when no message is open.
- A delta is provisional until its message closes and validates. Consumers may
  stream provisional data when their interface permits retraction or may wait
  for `take_completed()` when malformed output must never become visible.
- The parser exposes state for consumers to route messages and select their
  own failure policy. It does not execute tools or assemble conversations.

### Scope and lifecycle

`StreamParser` parses output framing only. It does not parse system prompts,
input messages, prefill text, or complete conversations.

A parser handles one contiguous engine-output segment. Its accepted forms are:

```text
OUTPUT_MESSAGE ("\n\n" OUTPUT_MESSAGE)* ["\n" WAIT]
WAIT
```

An empty segment and a segment ending at a clean message boundary are
syntactically complete when `finish()` is called, with `waiting == False`.
Whether such an engine stop is abnormal is consumer policy. A bare wait is
valid because memory stripping can leave a wait-only burst.

- Accepting `<|wait|>` sets `waiting` and makes the parser terminal.
- `finish()` at a clean boundary succeeds and makes the parser terminal.
- `finish()` inside a control-token prefix, header, payload, or separator
  raises `IncompleteMessageError` and makes the parser terminal.
- Any parsing failure makes the parser terminal.
- `push()` after a successful finish, and a second `finish()`, raise
  `ParserStateError`. This is caller misuse, not malformed model output.
- A failed parser cannot be reset or resumed. Recovery always uses a new
  `StreamParser` selected and driven by the consumer.
- If a harness splices input between model outputs, each contiguous engine
  output segment uses a new parser. Conversation assembly belongs to the
  harness.

### Strict canonical string grammar

The current text parser accepts exactly the string form emitted by
`render_str()` for output messages. It does not normalize whitespace.

```text
OUTPUT_MESSAGE := "<|out|>" HEADER "<|hdr|>" BODY "<|/out|>"
HEADER         := SP "type=" VALUE *(SP KEY "=" VALUE) SP
BODY           := SP                          ; empty payload
                | SP PAYLOAD SP               ; non-empty payload
KEY            := [A-Za-z0-9_]+
SP             := one ASCII space
```

- `type` is first and appears exactly once.
- Header fields have exactly one ASCII space between them.
- The parser consumes exactly one structural space after `<|hdr|>`.
- For a non-empty payload, it removes exactly one structural space before
  `<|/out|>` and preserves every additional leading or trailing payload space.
- The single space between `<|hdr|>` and `<|/out|>` represents an empty
  payload.
- Output messages are separated by exactly one blank line (`"\n\n"`).
- A final wait is attached to the preceding message with exactly one newline.
- Any content after a wait is a framing failure.
- Full control glyphs in payload text are structural in the partial string
  codec. An unexpected glyph therefore fails parsing. Payloads that spell
  control glyphs remain outside the string codec's unambiguous subset.

The machine keeps one candidate payload character until it sees the next
lexeme. At a close token, that character must be the structural trailing
space and is discarded. Otherwise it is emitted as payload. This preserves
payload whitespace without using `strip()` and makes chunk splitting
irrelevant.

### Header and message construction

Header parsing enforces the model's existing key and value grammars before
building a message:

- header keys match `[A-Za-z0-9_]+`;
- values are non-empty Unicode scalar strings containing no whitespace, `=`,
  `<|`, or `|>`;
- duplicate keys are rejected;
- `direction` is forbidden because `<|out|>` supplies it;
- known typed headers follow `type` before extras;
- `tool_call` requires exactly one `id` immediately after `type`;
- extras retain observed insertion order.

After the header terminator, `current_type` and an immutable
`current_headers` mapping are available. At message close, a cached explicit
dispatch performs exactly one Pydantic validation:

```python
_OUTPUT_TYPES = {
    "assistant": Assistant,
    "think": Think,
    "tool_call": ToolCall,
    "verifiable_answer": VerifiableAnswer,
}
```

Unknown output types use `GenericOutput`.

#### Tool-call payloads

A tool-call payload accepts semantically valid JSON rather than requiring the
exact compact byte spelling produced by `render_str()`:

- the top level is an object containing exactly `name` and `args`;
- top-level key order and insignificant JSON whitespace may vary;
- duplicate object keys are rejected at every nesting level;
- `NaN`, `Infinity`, and `-Infinity` are rejected;
- `name`, `args`, and the header `id` pass through `ToolCall` validation.

Invalid JSON or top-level structure raises `MessageValidationError` with
reason `INVALID_TOOL_PAYLOAD`. Other Pydantic failures use
`MESSAGE_VALIDATION`. No partial or invalid `ToolCall` enters `messages` or
the completed queue.

### Failure contract

The parser reports failures but never recovers from them:

```text
ApertusFormatError
└── StreamParseError
    ├── FramingError
    │   └── IncompleteMessageError
    └── MessageValidationError

ParserStateError(RuntimeError)
```

Every `StreamParseError` exposes:

```python
error.reason              # stable ParseFailure enum
error.parser              # exact terminal StreamParser that raised
error.offset              # absolute character offset in text mode
error.message_ordinal     # zero-based output-message position
error.state               # state at failure
error.raw_text            # complete uncommitted frame or fragment
error.unconsumed          # suffix beginning at the rejected source unit
error.__cause__           # JSON/Pydantic cause, when applicable
```

`error.parser` is the parser instance whose operation failed. In streaming
use, it is identical to the caller-owned parser. In batch use,
`StreamParser.parse(text)` creates the parser internally, so the exception is
the caller's route to its retained messages, pending queues, current frame,
and failure state. The reference is not a snapshot: the parser is already
terminal and cannot be mutated by further parsing.

The initial stable reasons are:

```text
UNEXPECTED_TEXT
UNEXPECTED_CONTROL
INVALID_SEPARATOR
INVALID_HEADER
DUPLICATE_HEADER
MISSING_TYPE
MISMATCHED_CLOSE
INVALID_TOOL_PAYLOAD
MESSAGE_VALIDATION
TRAILING_AFTER_WAIT
INCOMPLETE_CONTROL
INCOMPLETE_HEADER
INCOMPLETE_PAYLOAD
INCOMPLETE_SEPARATOR
```

On failure, the parser:

1. stops at the first rejected source unit;
2. preserves completed messages and queued deltas produced before it;
3. preserves the uncommitted raw fragment and unaccepted suffix;
4. attaches itself to the public exception as `error.parser`;
5. enters a permanent failed state;
6. performs no lookahead for a recovery boundary and no raw-content fallback.

The consumer may propagate the error, retain committed messages, use or
discard provisional deltas, show raw output, or feed a chosen suffix to a new
parser. Those actions are outside `StreamParser`.

### Lexeme boundary

The output machine consumes source-neutral lexemes from the text frontend:

```python
@dataclass(frozen=True, slots=True)
class SourceSpan:
    start: int
    end: int
    unit: Literal["character", "token"]

@dataclass(frozen=True, slots=True)
class TextLexeme:
    text: str
    span: SourceSpan

@dataclass(frozen=True, slots=True)
class ControlLexeme:
    token: ControlToken
    span: SourceSpan

Lexeme = TextLexeme | ControlLexeme
```

Frontends implement one internal protocol:

```python
class Lexer[InputT](Protocol):
    def push(self, value: InputT) -> Sequence[Lexeme]: ...
    def finish(self) -> Sequence[Lexeme]: ...
```

```text
TextLexer -> Lexeme stream -> OutputMachine -> StreamParser state
```

The facade retains the uncommitted raw source. The machine therefore uses
source spans to report failures without storing source-specific values in
each lexeme.

#### Implemented text frontend

The text lexer recognizes the complete library control-glyph inventory using
one precompiled longest-first literal alternation. It emits maximal text runs
and individual control lexemes with absolute character spans. Streaming keeps
only the longest suffix that is a prefix of a control glyph, allowing glyphs
to straddle arbitrary chunks without delaying unrelated text.

Only `OUT`, `OUT_END`, `HDR`, and `WAIT` are legal to the output machine.
System, input, and padding controls are recognized so they can produce a
precise `UNEXPECTED_CONTROL` failure.

### Internal architecture

```text
core/_lexeme.py       # source-neutral lexeme and span types
core/_scan.py         # TextLexer with control-prefix holdback
core/_machine.py      # strict output-only state machine
parsing/_stream.py    # public facade, queues, builder, lifecycle
```

The machine states cover expecting output, header, payload, separator,
waiting, finished, and failed. It mutates plain internal state and never runs
Pydantic validation inside the lexeme loop. The facade builds one typed model
at message close and owns cumulative and drainable state views.
