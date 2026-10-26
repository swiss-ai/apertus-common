"""Print the conformance rule table.

Run with ``make rules`` or ``uv run --locked python scripts/rules.py``.
"""

from __future__ import annotations

from apertus_common import rule_table


def main() -> None:
    table = rule_table()
    width = max(len(rule.id) for rule in table)
    header = f"{'rule id':{width}}  {'severity':8}  description"
    print(header)
    print("-" * len(header))
    for rule in table:
        print(f"{rule.id:{width}}  {rule.default_severity:8}  {rule.description}")
    print(f"\n{len(table)} rules")


if __name__ == "__main__":
    main()
