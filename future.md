# Future work

Considerations that are not planned yet. Committed scope and plans live in
[DEVELOPMENT.md](DEVELOPMENT.md).

## Expose claims through the OpenAI mapping

**Current behaviour (option A).** Serving adapters map output with
`MappingPolicy(omit_claims=True)`. Chat output omits `claim` messages, which
the profile treats as inert where the deployment does not grade them, and each
omission produces a `claim-omitted` diagnostic. Native token parsing keeps
every claim.

**Consideration (option C).** Project claims into an OpenAI extension field,
in the style of `reasoning_content`, such as a `claims` list on the assistant
message and its stream deltas. Graders and RL environments that call an
OpenAI-compatible endpoint could then read the verifiable answer without
native token parsing.

**When it is useful.** RL rollouts or evaluations that use vLLM's chat API
instead of the library's token parser, and grading that needs the claim
separately from the reply. Pipelines that parse tokens natively already
receive claims.

**What it involves:**

- A mapping option that keeps claims in `OpenAIAssistantMessage` and
  `OpenAIDelta`, with their order relative to reasoning and content defined.
- Restoring claims from the extension field in `from_openai`, so replayed
  prompts match the generated tokens.
- Adapter support: vLLM response and delta protocol fields, and the
  Transformers pipeline output.
- Client documentation: standard OpenAI clients ignore unknown fields.
