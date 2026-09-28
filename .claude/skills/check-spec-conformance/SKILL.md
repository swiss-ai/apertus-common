---
name: check-spec-conformance
description: Audit apertus-common against the Apertus format and profile, DEVELOPMENT.md scope, docs/design.md implemented contracts, and available tokenizer and conformance artifacts. Use for conformance reviews after implementation or upstream changes, or before a release; report divergences without editing code or golden files.
---

# check-spec-conformance

Verify the implementation against its sources and report findings. This skill
is report-only: do not edit code, documentation, or golden files, and do not
stage or commit changes.

## Baseline

- `DEVELOPMENT.md` owns scope, plans, exclusions, and open decisions.
- `docs/design.md` records implemented behavior. The current pilot has not yet
  been aligned with the current Apertus 2 profile; distinguish this known gap
  from regressions against implemented contracts.
- `README.md` and `AGENTS.md` identify upstream sources and local checkouts.
  Read both `spec.md` and `profile/apertus_2/apertus_2.md` from a compatible
  upstream revision. Do not assume the reference profile and Apertus 2 profile
  have identical message types or payloads.
- The tokenizer repository owns the vocabulary and control-token inventory.
  Discover the available artifact and branch rather than assuming an old path
  or branch is still current.
- Golden conformance files, when available, specify exact token IDs and parser
  results. Their packaging remains planned until implemented.

For current upstream checks, fetch refs without changing working trees. Never
pull, checkout, or reset the source repositories. Record the revisions used;
if fetching fails, report the local baseline and its freshness limitation.

## Audit

1. Read the baseline documents, format and profile, and available tokenizer
   inventory. Note unresolved scope decisions instead of treating them as
   requirements already adopted.
2. Inventory the model, codecs, parser, checker, adapters, tests, and fixtures.
   Mark each scoped capability Implemented, Partial, Missing, or Divergent.
3. Run the repository's `make test` target. Capture relevant failures without
   changing code or fixtures to make them pass.
4. Compare implementation against `docs/design.md`, then separately compare it
   against the target profile and development plan. Check framing, message and
   header grammar, call/result relationships, authority/order, parser failure
   behavior, adapter mappings, and documented loss boundaries.
5. Check tokenizer compatibility when an artifact is available. Reserved IDs
   must enter through typed structure; ordinary text must be encoded with
   special-token recognition disabled. Report absent token integration as a
   planned capability, not as an implemented guarantee.
6. Verify exclusions: no Jinja maintenance or generation, multimodal
   integration, tool execution, or harness memory/scheduling policy.
7. Classify findings as library bugs, planned gaps, upstream drift, stale
   fixtures, or documentation drift, with file/line and revision evidence.

## Report

Lead with the verdict and revisions checked. Summarize capability status,
itemize divergences with evidence, and propose follow-ups ordered by impact.
Do not silently resolve conflicts between upstream semantics and library
scope. Propose fixture changes explicitly rather than regenerating them.

If sources are unavailable or tests cannot run, state what was checked and
what remains unverified. Never claim conformance for unverified paths.
