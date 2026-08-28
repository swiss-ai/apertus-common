# apertus-common roadmap

This document owns implementation sequencing, status, and unresolved work.
[SPEC.md](../SPEC.md) defines the required library scope and public guarantees;
[design.md](design.md) records implementation architecture and settled design
decisions. Parser implementation details live in [parsing.md](parsing.md).

## Milestones

Milestone identifiers retain their capability-plan numbers. Rows are ordered
by actual delivery: completed work first, then active and future work.

| Milestone | Delivers | Exit criteria | Status |
|---|---|---|---|
| **M0 foundation** | `src/` layout, project tooling, typed message cast, `SystemPrompt`, `Conversation`, and `normalized()` | Model round-trip and normalization tests are green; public surface is pinned | done |
| **M4 format adapters** | Native JSON entry points with ingest flags, JSON-lines dataset convention, OpenAI messages mapping, and the [formats guide](formats.md) | JSON-lines corpus round-trips byte-stably; OpenAI adapter and public API fixtures pass | done |
| **M1 core string pipeline and streaming parser** | Wire IR, `render_str`, public model checker, source-neutral lexemes, text lexer, strict output machine, `StreamParser`, and `examples/parsing_examples.ipynb` | Rendering matches spec examples; checker fixtures cover each serving rule; parser split-invariance, strict-spacing, routing, terminal-failure, lifecycle, and notebook checks pass | done |
| **M2 tokens** | Tokenizer artifact binding, `encode`, and token-id frontend for `StreamParser` | `encode == tokenize(render_str)`; golden token-id files; text and token frontends are equivalent | open, artifact-gated by O3 |
| **M3 reference parsers and engine adapters** | Tool and reasoning parsers over `StreamParser` state, plus engine adapter conventions | Explicit consumer-policy tests on failures; engine adapter conformance fixtures | open |
| **M5 golden contract** | Packaged golden files and the full core conformance fixture matrix | Deterministic fixtures are consumable without importing the library | open |

## Current focus

With M0, M4, and M1 complete, the next milestones are token work (M2, gated
on the tokenizer artifact, see O3) and the reference tool and reasoning
parsers over `StreamParser` state (M3). The parser interface, grammar,
failure behavior, notebook outline, and test matrix are maintained in
[parsing.md](parsing.md).

Optional extensions are scoped in [SPEC.md §5](../SPEC.md#5-optional-later-extensions).
They do not expand the active milestones unless promoted here.

## Open questions

- **O1, string escaping:** the string form currently renders payload text
  verbatim. Content validation or escaping for full losslessness remains
  deferred until adversarial string-form fixtures require a decision. See D3.
- **O2, Jinja equivalence owner:** decide whether the diagnostic template
  belongs in this repository or the tokenizer artifact. See D8.
- **O3, tokenizer artifact consumption:** choose a local path, Hub reference,
  or hash-pinned distribution. The current `tokenizers/Apertus_2` artifact on
  `v2-on-pr24` includes multimodal controls but not the conversation framing
  glyphs such as `<|sys|>`, `<|in|>`, `<|out|>`, `<|hdr|>`, and `<|wait|>`.
  Token-path work is blocked until the spec and artifact converge.
- **O5, tool-call id echo:** the v1 format spec requires `NAME:COUNTER`, while
  the #765 serving decision accepts `[A-Za-z0-9_:.-]{1,64}` byte-for-byte.
  Raise the divergence upstream. The interim model remains permissive and the
  checker warns for non-canonical ids. See D13.
- **O6, header values:** escaping for externally influenced header values is
  unsettled. The interim model uses a conservative grammar and rejects values
  outside it. See D12.

Resolved questions belong in the corresponding design decision rather than
this list. The JSON-lines dataset convention, formerly O4, is settled in D18.
