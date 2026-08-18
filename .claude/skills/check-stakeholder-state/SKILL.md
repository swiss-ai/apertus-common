---
name: check-stakeholder-state
description: Check the Apertus 2 stakeholder sources — the format spec repo (apertus-2-format-spec-proposal), the tokenizer repo (apertus-omni-tokenizer), and tracking issues swiss-ai/apertus-program#856/#765 — for new commits, branches, PRs, and comments since the last check; summarize them with their implications for apertus-common and flag needed SPEC.md/AGENTS.md updates. Use when the user asks for stakeholder/upstream status, what changed in the spec or tokenizer, or before starting a new piece of implementation work.
---

# check-stakeholder-state

Report what changed upstream and what it means for this library. This skill is
**report-only**: never edit SPEC.md, AGENTS.md, code, or the stakeholder repos
as part of it — recommend edits instead (AGENTS.md convention: in case of
deviations, inform the user).

## Sources

| Source | Location | Watch for |
|---|---|---|
| Format spec | local `/Users/rkre/swissai_repos/apertus-2-format-spec-proposal`, remote [swiss-ai/apertus-2-format-spec-proposal](https://github.com/swiss-ai/apertus-2-format-spec-proposal) | commits on the current revision branch (`v1`, see AGENTS.md) and `main`, new `vN` revision branches, open/updated/merged PRs, `spec.md` diffs |
| Tokenizer | local `/Users/rkre/swissai_repos/apertus-omni-tokenizer`, remote [swiss-ai/apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer) | commits on the checked-out branch (see AGENTS.md; currently `v2-on-pr24`) and `main`, PRs, changes under `tokenizers/Apertus_2/`, `chat_templates/`, `omnitok/apertus.py` |
| Tracking issues | [swiss-ai/apertus-program#856](https://github.com/swiss-ai/apertus-program/issues/856) and [#765](https://github.com/swiss-ai/apertus-program/issues/765) | new comments, state/milestone/assignee changes, newly cross-referenced issues or PRs |

Baseline state lives in `.claude/skills/check-stakeholder-state/state.json`:

```json
{
  "last_run": "ISO-8601 UTC",
  "spec_repo":      {"branches": {"origin/v1": "sha", "origin/main": "sha"}, "remote_branches": ["..."]},
  "tokenizer_repo": {"branches": {"origin/v2-on-pr24": "sha", "origin/main": "sha"}, "remote_branches": ["..."]},
  "issues": {"856": {"comment_count": 0, "updated_at": "..."}, "765": {"comment_count": 0, "updated_at": "..."}}
}
```

On the first run (no state file) the baseline is what SPEC.md and AGENTS.md
record: report the current upstream state against those documents, then write
the state file.

## Steps

1. Read `state.json` (if present), plus SPEC.md and AGENTS.md for the recorded
   assumptions (branch names, settled serving rules, tokenizer branch).
2. Fetch, without touching working trees — `git fetch` only, never
   pull/checkout/reset (the user may have work in progress there):
   `git -C <local> fetch --all --prune` for both repos.
3. Diff commits and branches: `git -C <local> log --oneline LAST_SHA..origin/BRANCH`
   for each tracked branch (on first run: `git log --oneline -15 origin/BRANCH`);
   compare `git branch -r` against the stored `remote_branches` to spot new
   revision branches. For the spec repo, read the actual content change:
   `git -C <local> diff LAST_SHA..origin/v1 -- spec.md`.
4. PRs: `gh pr list --repo <owner/repo> --state all --limit 20 --json number,title,state,updatedAt,headRefName,baseRefName,url`
   for both repos; keep those with `updatedAt > last_run`. Read the bodies of
   relevant ones (`gh pr view N --repo ...`).
5. Issues: for 856 and 765,
   `gh issue view N --repo swiss-ai/apertus-program --json state,title,milestone,updatedAt,comments`
   — new comments are those beyond the stored `comment_count` (or after
   `last_run`). Read the new comment bodies in full; summarize substance, not
   titles.
6. Classify every change by its effect on apertus-common:
   - **Format change** (spec.md, revision branches) → which SPEC.md §3/§4
     requirement or encoder/parser/adapter behavior it touches; whether golden
     files would change.
   - **Tokenizer change** (artifact, recipe, chat templates) → control-token
     ids/vocabulary compatibility, whether the pinned branch in AGENTS.md/SPEC.md
     is stale.
   - **Decision/process change** (issue comments, PRs merged) → scope changes
     for SPEC.md, new settled rules, ownership/timeline updates.
   - **No impact** — say so, don't pad.
7. Deliver the report (format below).
8. Update `state.json` with the new SHAs, branch lists, comment counts, and
   `last_run` (`date -u +%Y-%m-%dT%H:%M:%SZ`).

## Report format

Lead with a one-line verdict (e.g. "2 upstream changes, 1 affects the encoder"
or "No changes since <date>"). Then:

- **What changed** — per source, only sources with changes; link commits/PRs/
  comments.
- **Implications for apertus-common** — concrete: which components, which
  SPEC.md sections, whether golden files or the tokenizer pin are affected.
- **Suggested follow-ups** — doc edits (SPEC.md/AGENTS.md), code work, or
  questions to raise on #765/#856. Do not apply them.

## Failure handling

- `gh` unauthenticated or network down: report which sources were skipped and
  what was checked locally; don't fail silently.
- Local checkout's branch differs from what AGENTS.md records: flag it in the
  report (docs or checkout is stale).
- Never update `state.json` for a source that could not be checked.
