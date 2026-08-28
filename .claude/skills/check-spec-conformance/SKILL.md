---
name: check-spec-conformance
description: Audit apertus-common against its normative sources, including the Apertus 2 format spec (v1 branch of apertus-2-format-spec-proposal), this repo's SPEC.md, the golden conformance files, and the tokenizer artifact from apertus-omni-tokenizer, and report divergences without changing golden files or code. Use when the user asks whether the library conforms to the spec, after implementing a component, after a spec or tokenizer update, or before a release/PR.
---

# check-spec-conformance

Verify the implementation against what is normative, and report. This skill is
**report-only**: never edit code, and **never regenerate or edit golden files**
because they are the conformance contract; changing them is a deliberate user
decision, not a fix for a red test.

## Normative baseline

| Authority | Source | Governs |
|---|---|---|
| Format | `git -C /Users/rkre/swissai_repos/apertus-2-format-spec-proposal show origin/v1:spec.md` (read from `origin/v1`, not the working tree; the checkout may be on another branch or dirty) | token-level format: framing, control tokens, bursts/halt states, input/output types, order & authority, multimodal, system prompt, pretraining |
| Scope & API | `SPEC.md` in this repo | required components (§3), serving rules (§4), non-goals (§7) |
| Contract | golden test files in this repo (location per SPEC.md §3.5; may not exist yet) | exact token ids + parse events per conversation |
| Vocabulary | tokenizer artifact `tokenizers/Apertus_2/` in `/Users/rkre/swissai_repos/apertus-omni-tokenizer` (branch per AGENTS.md, currently `v2-on-pr24`) | control-token ids, special-token inventory, multimodal meta |

Run `git -C <repo> fetch origin` first so `origin/v1` and the tokenizer branch
are current; fetch only, never pull/checkout/reset.

## Steps

1. Load the baseline: the v1 `spec.md`, this repo's `SPEC.md`, and the
   tokenizer artifact's special-token inventory (e.g. `tokenizer_config.json`
   / `special_tokens_map.json` / added-tokens in `tokenizer.json` under
   `tokenizers/Apertus_2/`).
2. Inventory the implementation: package layout, which SPEC.md §3 components
   exist (typed model, codecs, incremental parser, conformance checker,
   OpenAI adapter, and golden files), which optional SPEC.md §5 components
   exist, and where the tests live. If the package does not exist yet, skip to
   the report and state the gap against SPEC.md §3; that is a valid result,
   not a failure.
3. Run the tests: `uv run pytest` (golden-file tests first if selectable).
   Capture failures verbatim; never "fix" a failure by touching golden files.
4. Walk the requirement checklist and mark each **Implemented / Partial /
   Missing / Divergent**, with file:line evidence:
   - every SPEC.md §3 component and §7 non-goal (non-goals must stay absent,
     e.g. no working Jinja render path, no encode bypass for untrusted text);
   - every SPEC.md §4 serving rule: `<|wait|>` as the only stop token,
     decode-time suppression of input framing, special-tokens-disabled
     encoding of untrusted text, fixed `tool_call` payload grammar
     `{"name": ..., "args": ...}`, id echo `[A-Za-z0-9_:.-]{1,64}` with
     `NAME:COUNTER` as the model-written convention, canonical input order,
     unknown vs known-but-unconsumed output types, and the rank-4 default for
     unranked types;
   - parser guarantees (SPEC.md §3.3): strict first failure, retained terminal
     parser state, and recovery selected only by the consumer;
   - adapter guarantees (SPEC.md §3.4): role mapping incl. mid-conversation
     `system` → `harness`, think → `reasoning_content` only,
     `finish_reason` = `stop`/`tool_calls`/`length`, assistant prefill;
   - loss-mask rules (SPEC.md §5.1): input framing never a target, output
     framing a target where the message carries loss, think/annotation loss a
     recipe choice.
5. Check vocabulary compatibility: every control token the library references
   resolves to the id the tokenizer artifact defines; flag tokens the library
   hard-codes, misses, or that the artifact lacks.
6. For each divergence, classify the likely cause; do not guess silently;
   say which evidence supports it:
   - **library bug** (spec and golden file agree, code doesn't);
   - **library lags upstream** (spec on `origin/v1` changed; suggest running
     `/check-stakeholder-state` for the full delta);
   - **stale golden file** (code matches the current spec, file matches an
     older one; regeneration is a user decision to propose, never perform);
   - **doc drift** (SPEC.md or AGENTS.md no longer matches the spec/tokenizer).
7. Deliver the report. Do not stage, commit, or edit anything.

## Report format

Lead with a one-line verdict ("Conformant", "N divergences (M likely library
bugs)", or "Package not started: core SPEC.md §3 scope is absent"). Then:

- **Checklist:** the §3/§4 table with statuses; collapse long
  fully-conformant stretches to one line, itemize everything else.
- **Divergences:** each with evidence (file:line, failing test output, spec
  section) and its classification from step 6.
- **Suggested follow-ups:** ordered by severity: code fixes, proposed golden
  regeneration (as a question to the user), doc updates, questions for
  #765/#856. Do not apply them.

## Failure handling

- Spec repo or tokenizer repo unreachable/unfetchable: fall back to the local
  checkout, and say so in the report (results may be stale).
- Tests unrunnable (no pytest, broken env): report that as its own finding;
  static checks still count.
- Ambiguity between format spec and SPEC.md: format spec wins on format,
  SPEC.md on scope/API (SPEC.md §2); flag the conflict for the user instead
  of resolving it in either document.
