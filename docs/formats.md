# Native JSON and OpenAI message formats

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

## Shared ingest controls

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

## Native JSON

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

### JSON-lines datasets

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

## OpenAI messages into the model

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
| `user` `text`, `image_url`, `input_audio` parts | `TextPart`, `ImagePart`, `AudioPart` |
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

The adapter ignores OpenAI `name` fields and unknown keys. Media `detail` and
audio `format` are not retained. Default normalization can move a later system
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

## Model into OpenAI messages

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
| `User` | `user`, including text/image/audio parts |
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

Round trips are therefore exact only within the adapter subset and up to these
normalizations: `developer` returns as `system`; assistant `content` is always
present and may be `None`; tool arguments are re-serialized compactly; system
identity, behavior, and environment lose their tag boundaries; ignored OpenAI
keys and media options do not return.

```python
openai_request = to_openai(conversation)
messages = openai_request["messages"]
```

## Errors

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
