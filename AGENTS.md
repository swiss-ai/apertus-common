# apertus-common agent guide

`CLAUDE.md` is a symlink to this file. Edit `AGENTS.md`, never both.

## Project references

`apertus-common` is a typed reference library for the Apertus conversation
format. The text-only implementation targets the Apertus 2 profile; external rollout
gates are tracked in DEVELOPMENT.md.

- Read `DEVELOPMENT.md`, when available, before implementing changes. This
  ignored, local-only file owns scope, plans, open decisions, and module/API
  conventions; it is not included in fresh clones.
- [docs/design.md](docs/design.md) records implemented behavior. Keep it in
  sync with code; future proposals belong in `DEVELOPMENT.md`.
- [README.md](README.md) provides the overview, next tasks, upstream links,
  and canonical development commands.
- Format and profile repository: `/Users/rkre/swissai_repos/apertus-2-format-spec-proposal`.
  Read its `spec.md` and `profile/apertus_2/apertus_2.md` from upstream `v1`
  (`origin/v1` in the local checkout; the local `v1` branch may be stale).
  The implemented revision is recorded in `docs/design.md`. This repository is a
  read-only source of truth; raise conflicts instead of silently deviating.
- Tokenizer repository: `/Users/rkre/swissai_repos/apertus-omni-tokenizer`.
  The Apertus 2 instruct artifact is `tokenizers/Apertus_2_instruct` from PR #35
  until merged. Consume it through `load_encoding`; never vendor or hand-edit
  tokenizer files here.
- Framework integration checkouts (Transformers, vLLM):
  `/Users/rkre/swissai_repos/apertus2-integration-work/`; status and open
  findings are in DEVELOPMENT.md.
- Core interface references: [openai-harmony](https://github.com/openai/harmony),
  local checkout `/Users/rkre/swissai_repos/harmony`, and
  [mistral-common](https://github.com/mistralai/mistral-common), local checkout
  `/Users/rkre/PycharmProjects/mistral-common`.
- Local Swiss AI vLLM checkout: `/Users/rkre/swissai_repos/vllm_swissai`.
- Program tracking issues are listed in the local `DEVELOPMENT.md`, when available.

## Working conventions

- Use the `Makefile` for installation, formatting, linting, type checking,
  repository checks, tests, and the full CI gate. Commands are maintained in
  the [README development section](README.md#development).
- Python >= 3.13; use `uv` for dependencies and execution.
- Follow the module and API conventions in local `DEVELOPMENT.md`, when available.
- Keep documentation and comments concise and free of em dashes.
- No Jinja maintenance or generation. Multimodal integration is out of scope.
