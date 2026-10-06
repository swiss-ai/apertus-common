# apertus-common agent guide

`CLAUDE.md` is a symlink to this file. Edit `AGENTS.md`, never both.

## Project references

`apertus-common` is a typed reference library for the Apertus conversation
format. The text-only implementation targets the Apertus 2 profile; external rollout
gates are tracked in DEVELOPMENT.md.

- Read [DEVELOPMENT.md](DEVELOPMENT.md) before implementing changes. It owns
  scope, plans, open decisions, and module/API conventions.
- [docs/design.md](docs/design.md) records implemented behavior. Keep it in
  sync with code; future proposals belong in `DEVELOPMENT.md`.
- [README.md](README.md) provides the overview, next tasks, upstream links,
  and canonical development commands.
- Format and profile repository: `/Users/rkre/swissai_repos/apertus-2-format-spec-proposal`.
  Read its `spec.md` and `profile/apertus_2/apertus_2.md`. This repository is a
  read-only source of truth; raise conflicts instead of silently deviating.
- Tokenizer repository: `/Users/rkre/swissai_repos/apertus-omni-tokenizer`.
  Consume its artifact through the planned integration; never vendor or
  hand-edit tokenizer files here.
- Core interface reference: [openai-harmony](https://github.com/openai/harmony),
  local checkout `/Users/rkre/swissai_repos/harmony`.
- Local Swiss AI vLLM checkout: `/Users/rkre/swissai_repos/vllm_swissai`.

## Working conventions

- Use the `Makefile` for installation, formatting, linting, type checking,
  repository checks, tests, and the full CI gate. Commands are maintained in
  the [README development section](README.md#development).
- Python >= 3.13; use `uv` for dependencies and execution.
- Follow [module and API conventions](DEVELOPMENT.md#module-and-api-conventions).
- Keep documentation and comments concise and free of em dashes.
- No Jinja maintenance or generation. Multimodal integration is out of scope.
