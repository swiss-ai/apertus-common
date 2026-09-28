# Development Planning

This is the single development reference for scope, plans, open decisions, and
conventions. [docs/design.md](docs/design.md) records implemented behavior;
[README.md](README.md) provides the overview and next tasks.

State: An initial Apertus 2 profile is available as the implementation baseline.
The profile will continue to evolve. As of September 25, 2026, this library
contains an older pilot implementation.

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

In parallel, update the Apertus 2 tokenizer in its own repository to support
the new special tokens. Target Hugging Face `tokenizers` v1 compatibility,
verify the APIs required for encoding and incremental decoding, and benchmark
representative Apertus workloads. Tokenizer artifact changes remain owned by
the tokenizer repository.

## References

- Local specification repository: `/Users/rkre/swissai_repos/apertus-2-format-spec-proposal`
- Local tokenizer repository: `/Users/rkre/swissai_repos/apertus-omni-tokenizer`
- Core interface reference: [`openai-harmony`](https://github.com/openai/harmony),
  with local checkout at `/Users/rkre/swissai_repos/harmony`.

## Stage 1 - Define scope and interfaces

The library provides reusable format operations for Apertus 2 pipelines while
leaving harness responsibilities to its consumers. Target use cases include:

- Pretraining, SFT, and RL infrastructure, such as Megatron and NeMo RL.
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
Training examples with malformed headers or broken framing would require a
separate raw-data path, whose scope remains to be defined.

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

## Next work and open decisions

The current text-only pilot provides typed models, conformance checking,
string rendering, native JSON/OpenAI adapters, and experimental output parsing.
The current Apertus 2 profile changes message names, header grammar, tool-call
identity, and system-prompt representation. Profile alignment remains pending.

1. Finalize capability boundaries and interfaces, including compatibility with
   existing pilot APIs and datasets. Decide whether full-transcript parsing,
   training loss masks, packing, and malformed raw training examples belong in
   scope. Assign ownership for any framework-specific adapters.
2. Adapt the typed model, checker, renderer, parser, and OpenAI mappers to an
   identified revision of the format and Apertus 2 profile. Resolve header
   escaping, string round-trip guarantees, and external tool-ID mapping against
   those references rather than carrying forward pilot assumptions.
3. Integrate the tokenizer artifact and implement token encoding and an
   incremental token frontend. Choose the artifact source and version pinning;
   verify its control-token inventory and tokenizer v1 compatibility.
4. Build reference tool/reasoning consumers and lightweight serving adapters.
   Define generation-prefix and response behavior, including stop, tool-call,
   and length finish reasons. Integrations must use the shared parser state.
5. Package deterministic conformance fixtures for strings, token IDs, parser
   results, and malformed input, consumable without importing the library.
   Validate end-to-end integration and measure performance before optimizing.

For token encoding, ordinary text must never produce special-token IDs.
Adjacent text atoms must be encoded as one maximal text run, since separately
encoding their pieces can change tokenization. Text and token parser frontends
should drive the same output machine. These are implementation requirements,
not capabilities currently provided by the pilot.

The library does not execute tools, schedule work, manage conversation memory,
or implement an inference engine. Tokenizer artifacts remain maintained in the
separate tokenizer repository. Jinja and multimodal integration are excluded.

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
