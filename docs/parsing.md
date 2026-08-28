# Streaming output parsing design

The serving reverse path is an output-only `StreamParser`. It consumes one
contiguous model-output segment, exposes provisional payload deltas and
completed typed output messages, and stops at the first parsing failure. It
never chooses a recovery policy. Full transcript parsing (`parse_str`) remains
an optional later extension, and `check(conversation)` remains model-only.

**Status: experimental.** The text parser is implemented and tested, but its
public API and error contract may change as tokenizer and serving integrations
land. There is no public token-ID parser yet.

Companion to [design.md](design.md) (D5, D7, D9, D17).

## Public interface

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
- There is one public parser class. Tool parsers, reasoning parsers, harness
  loops, and engine adapters consume this state and select their own failure
  policy. They do not subclass or reimplement the parser.

## Scope and lifecycle

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

## Strict canonical string grammar

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

## Header and message construction

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

### Tool-call payloads

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

## Failure contract

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

## Lexeme boundary

The implemented output machine consumes source-neutral lexemes. This boundary
was established with the text frontend so a future token frontend can reuse
the same state machine:

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
TextLexer  ──┐
             ├──> Lexeme stream ──> OutputMachine ──> StreamParser state
TokenLexer ──┘  (planned)
```

The facade retains the uncommitted raw source. The machine therefore uses
source spans to report failures without storing source-specific values in
each lexeme.

### Implemented text frontend

The text lexer recognizes the complete library control-glyph inventory using
one precompiled longest-first literal alternation. It emits maximal text runs
and individual control lexemes with absolute character spans. Streaming keeps
only the longest suffix that is a prefix of a control glyph, allowing glyphs
to straddle arbitrary chunks without delaying unrelated text.

Only `OUT`, `OUT_END`, `HDR`, and `WAIT` are legal to the output machine.
System, input, padding, media, and media-expansion controls are recognized so
they can produce a precise `UNEXPECTED_CONTROL` failure.

### Planned token frontend

No token frontend or `StreamParser.from_tokenizer()` constructor is public
today. The intended token mode has these requirements:

- Input mode is fixed for the parser's lifetime; text and IDs never mix.
- Registered special IDs produce `ControlLexeme` values directly.
- Ordinary IDs pass through a tokenizer-aware incremental decoder and produce
  `TextLexeme` values.
- Ordinary IDs that decode to a control glyph remain ordinary payload text.
- Token-mode spans use token offsets, and errors retain raw token IDs plus
  decoded diagnostic text.
- The exact incremental ordinary-token decoder remains gated on the tokenizer
  artifact and cannot assume that each token decodes independently.

The token frontend will implement the same lexeme protocol. It will not replace
or modify the output machine.

## Internal architecture

```text
core/_lexeme.py       # source-neutral lexeme and span types
core/_scan.py         # TextLexer with control-prefix holdback
core/_machine.py      # strict output-only state machine
parsing/_stream.py    # public facade, queues, builder, lifecycle
tokenizer/_scan.py    # planned TokenLexer, not present
```

The machine states cover expecting output, header, payload, separator,
waiting, finished, and failed. It mutates plain internal state and never runs
Pydantic validation inside the lexeme loop. The facade builds one typed model
at message close and owns cumulative and drainable state views.

## Performance strategy

1. The text scanner uses a compiled literal alternation and slices maximal
   text runs rather than stepping through every character in Python.
2. The machine performs Python work per lexeme and message boundary.
3. Pydantic validation runs once per completed message.
4. `messages` retains one reference per completed message. The completed and
   delta queues retain only entries not yet drained.
5. No zero-allocation claim is made. Lexemes, slices, deltas, queue entries,
   and validation necessarily allocate.
6. `scripts/bench_parse.py` records scanning and end-to-end throughput. A
   regression budget will be set from measured implementation baselines rather
   than from an unrelated primitive such as `str.count`.

## Documentation notebook

`examples/parsing_examples.ipynb` is the focused, executable guide to the
experimental parsing surface. It uses only public imports, runs without network
access, and contains assertions so examples fail visibly when the contract
changes. Internal modules remain private; architecture internals are explained
with diagrams and observed public behavior rather than imports from
underscore-prefixed modules.

The notebook is organized as follows:

1. **Interface and architecture overview.** Briefly explain the output-only
   scope, strict canonical grammar, provisional versus committed data,
   consumer-owned recovery, and the
   `TextLexer -> Lexeme -> OutputMachine -> StreamParser` flow.
2. **Streaming mode.** Feed the same rendered output using deliberately uneven
   chunks, including chunks that split control glyphs and cross message types.
   Drain `PayloadDelta` and completed-message queues, then inspect
   `current_type`, `current_headers`, `current_payload`, `messages`, and
   `waiting`.
3. **Batch mode.** Parse the same text with `StreamParser.parse()`, compare its
   typed messages with the streaming result, and show that batch uses the same
   parser path.
4. **Failure taxonomy.** Trigger and catch representative failures from each
   layer: `FramingError` for canonical grammar or header failures,
   `IncompleteMessageError` at end of input, `MessageValidationError` for
   invalid tool JSON and Pydantic model construction, and `ParserStateError`
   for caller misuse. Show `ParseFailure`, source offsets, message ordinals,
   parser states, and chained JSON/Pydantic causes.
5. **Raw failure data and consumer recovery.** Demonstrate
   `error.parser is parser`, then inspect `error.raw_text`,
   `error.unconsumed`, retained committed messages, provisional deltas, and
   the current frame. Show an explicitly consumer-selected recovery by
   constructing a new parser and feeding a chosen suffix. The example must not
   imply that the parser resynchronizes or repairs output automatically.
6. **Inner workings.** Briefly illustrate control-prefix holdback,
   source-neutral lexemes, strict state transitions, the one-character payload
   holdback used to preserve boundary whitespace, construction of a typed
   message only at close, and permanent terminal states. A split-invariance
   example compares whole-input, character-at-a-time, and uneven-chunk feeds.

The README links the notebook and includes its Jupyter launch command.
`scripts/check_notebooks.py` checks its code cells with the other examples,
and the complete repository gate covers its public imports and API usage.

## Testing

- Split invariance: character, random-chunk, and whole-input text feeds produce
  identical messages, failures, and concatenated deltas.
- Cross-boundary push: one chunk containing several output types preserves
  every delta with the correct ordinal and type.
- Reconstruction: feeding rendered output-only conversations reproduces the
  typed output messages on the unambiguous string subset.
- Strict spacing: empty payloads, one-space payloads, leading and trailing
  whitespace, embedded newlines, message separators, and wait attachment.
- Header grammar: missing or repeated type, duplicate keys, typed-key order,
  forbidden direction, invalid values, and preserved extras.
- Tool JSON: alternate key order and whitespace accepted; duplicate keys,
  non-finite numbers, extra or missing keys, and invalid models rejected.
- Output scope: system, input, padding, and media controls fail immediately.
- Failure state: exact reason, offset, ordinal, raw fragment, unconsumed suffix,
  chained cause, `error.parser is parser`, retained prior state, and permanent
  terminal behavior.
- Lifecycle: bare wait, clean finish without wait, incomplete finish, content
  after wait, push after finish, and repeated finish.
- Batch equivalence: `StreamParser.parse(text)` matches push then finish on a
  fresh parser; on failure, `error.parser` exposes that terminal batch parser.
- Queue semantics: `messages` remains cumulative while `take_completed()` and
  `take_deltas()` drain only pending entries.
- Planned tokenizer integration must add equivalence tests showing that token
  and text lexemes drive identical machine outcomes.

## Optional later extensions

- Strict full-transcript `parse_str(text) -> Conversation` may reuse the text
  lexer, header grammar, and builders. It is a separate consumer with system,
  input, conversation-order, and complete-model responsibilities.
- Token `decode` may use the planned token lexer and a transcript consumer.
- `check` over strings or token IDs may translate parser exceptions into
  framing findings before running model conformance checks.

These extensions share framing primitives. They do not expand the current
`StreamParser` beyond output-only parsing.

## Implemented text-parser components

- Parser exceptions and the `ParseFailure` enum.
- Source-neutral lexemes and character spans in `core/_lexeme.py`.
- `TextLexer` with control-prefix holdback and split-invariance tests.
- A strict output-only state machine for whitespace, headers, payloads,
  separators, waits, and terminal failures.
- The public `StreamParser` facade with typed builders, queues, lifecycle, and
  batch parsing.
- Malformed-input, reconstruction, routing, lifecycle, and performance checks.
- The executable parsing notebook and synchronized public documentation.
