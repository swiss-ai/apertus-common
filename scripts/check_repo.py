"""Repo invariant checks for developer tooling and CI.

Run with ``uv run python scripts/check_repo.py``. Exits non-zero on the
first failed invariant group and prints every violation found.

Current invariants:
- every declared rule id is unique across all builtin checkers
- every rule id follows the ``family/rule`` kebab-case grammar
- every rule has a non-empty description
"""

from __future__ import annotations

import sys
from collections import Counter

from apertus_common import checkers_for, rule_table
from apertus_common.checker._report import RULE_ID_RE


def check_rules() -> list[str]:
    problems: list[str] = []
    table = rule_table()

    counts = Counter(rule.id for rule in table)
    for rule_id, count in sorted(counts.items()):
        if count > 1:
            problems.append(f"duplicate rule id: {rule_id!r} declared {count} times")

    for rule in table:
        if RULE_ID_RE.fullmatch(rule.id) is None:
            problems.append(
                f"rule id {rule.id!r} violates the family/rule kebab-case grammar"
            )
        if not rule.description.strip():
            problems.append(f"rule {rule.id!r} has an empty description")

    for checker in checkers_for("full", training=True):
        if not checker.rules:
            problems.append(f"{type(checker).__name__} declares no rules")

    return problems


def main() -> int:
    problems = check_rules()
    if problems:
        print(f"repo check FAILED ({len(problems)} problem(s)):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"repo check OK: {len(rule_table())} rules, ids unique and well-formed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
