# apertus-common agent guide

Guidance for coding agents (and humans) working in this repository.
`CLAUDE.md` is a symlink to this file. Edit `AGENTS.md`, never both.

## What this repo is

`apertus-common` is the typed, ground-truth reference library for the
Apertus 2 conversation format, in the spirit of `mistral-common` and
`openai-harmony`.

Three references govern the work (full details and remotes
in [SPEC.md](SPEC.md) §2):

- **[SPEC.md](SPEC.md):** this repository's exact spec: scope, required
  components, serving rules, non-goals. Read it before implementing anything.
- **Format spec:** `spec.md` on the **`v1` branch** of
  [swiss-ai/apertus-2-format-spec-proposal](https://github.com/swiss-ai/apertus-2-format-spec-proposal)
  (local: `/Users/rkre/swissai_repos/apertus-2-format-spec-proposal`): the
  token-level format this library implements.
- **Tokenizer artifact:**
  [swiss-ai/apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer)
  (local: `/Users/rkre/swissai_repos/apertus-omni-tokenizer`, currently on
  branch `v2-on-pr24`) builds the Apertus 2 transformers tokenizer including
  multimodal meta information; this library consumes it and never hand-edits
  or vendors tokenizer files.

## Project context

Apertus 2 is the Swiss AI initiative's long-term model. Its conversation
format is message-based (Harmony-inspired) rather than turn-based.

Tracked in the swiss-ai/apertus-program project:

- [#856: Apertus 2 format reference library and conformance pilot](https://github.com/swiss-ai/apertus-program/issues/856)
  (milestone v1.6): scoping issue for this library and the end-to-end pilot;
  its "Library scope" section is the source of SPEC.md §3.
- [#765: Apertus 2 chat template format spec proposal](https://github.com/swiss-ai/apertus-program/issues/765):
  the format discussion; its review threads settled the serving rules and
  adapter decisions recorded in SPEC.md §4 and §3.4.

## Development

- Use the repository `Makefile` for installation, tests, formatting, linting,
  type checking, repository checks, and the complete CI gate. The canonical
  commands and descriptions are maintained in the
  [README development section](README.md#development); do not duplicate them
  here.
- Python ≥ 3.13 (`.venv/` exists); use `uv` for dependency management
  (`uv add`, `uv add --dev`, `uv run`).

## Conventions for agents

- [CONVENTIONS.md](CONVENTIONS.md) defines the module structure, naming,
  and public-API re-export rules; follow it when adding or moving code.
- Documentation and comments must not contain em dashes.
- Keep documentation and comments as concise as their purpose permits.
- The format spec repo is a **read-only source of truth**: never "fix" a
  format question by deviating here. In case of issues or deviations inform the user.
- Keep this library in sync with the state of the
  apertus-spec repo and be compatible with the apertus tokenizer artifact.
