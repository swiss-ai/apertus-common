---
name: check-stakeholder-state
description: Check the Apertus format/profile and tokenizer repositories and swiss-ai/apertus-program issues 856 and 765 for upstream changes and their implications for apertus-common. Use for stakeholder status or upstream-change reviews; suggest updates to DEVELOPMENT.md, docs/design.md, README.md, or AGENTS.md without applying them.
---

# check-stakeholder-state

Report upstream changes and their implications. Do not edit project documents,
code, or upstream working trees as part of this report. The only persistent
output is the check baseline in this skill's `state.json`.

## Sources and baseline

Read `README.md`, `DEVELOPMENT.md`, `docs/design.md`, and `AGENTS.md` for source
locations, development scope, and implemented behavior.

| Source | Location | Watch for |
|---|---|---|
| Format and profile | `/Users/rkre/swissai_repos/apertus-2-format-spec-proposal`; [remote](https://github.com/swiss-ai/apertus-2-format-spec-proposal) | relevant revision/profile branches, `main`, PRs, `spec.md`, and `profile/apertus_2/` changes |
| Tokenizer | `/Users/rkre/swissai_repos/apertus-omni-tokenizer`; [remote](https://github.com/swiss-ai/apertus-omni-tokenizer) | current and target branches, PRs, tokenizer artifacts and build recipes |
| Tracking issues | [856](https://github.com/swiss-ai/apertus-program/issues/856), [765](https://github.com/swiss-ai/apertus-program/issues/765) | new comments, status changes, and linked issues or PRs |

Discover the relevant branches and artifact paths from the recorded references
and local repositories. Do not assume an older branch or artifact path remains
current. Keep the shared format and model profile on compatible revisions.

The baseline file is `.claude/skills/check-stakeholder-state/state.json`.
Preserve its `last_run`, per-repository `branches` maps of refs to SHAs,
`remote_branches` lists, and per-issue `comment_count` and `updated_at` fields.
On the first run, compare upstream state with the project documents and record
an initial baseline.

## Check

1. Read the prior state and project documents.
2. Fetch refs for both repositories without touching working trees. Never
   pull, checkout, or reset; user work may be present.
3. Compare tracked branch SHAs and remote branch lists with the baseline.
   Read actual changes to the shared format and Apertus 2 profile, not just
   commit titles. Record the revisions used.
4. Inspect relevant PRs using `gh pr list` and `gh pr view`, filtering by the
   prior check time when available.
5. Read new issue comments and status changes with `gh issue view` for issues
   856 and 765. Summarize decisions and open questions, not just titles.
6. Classify effects on development scope, implemented contracts, tokenizer
   compatibility, adapters, tests, and fixtures. Distinguish planned profile
   alignment from newly introduced discrepancies.
7. Report findings, then update the baseline only for successfully checked
   sources, preserving prior state for sources that could not be checked.

## Report

Lead with a brief verdict. Link the relevant commits, PRs, or comments; explain
which components are affected and suggest follow-ups. Scope and plans belong
in `DEVELOPMENT.md`, implemented behavior in `docs/design.md`, overview and next
tasks in `README.md`, and agent entry points in `AGENTS.md`.

Report unavailable sources, network/authentication failures, and differences
between documented targets and local checkouts. Use local evidence when
necessary and make its freshness limitation explicit.
