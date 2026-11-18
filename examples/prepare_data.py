"""Demonstrate bounded checking, encoding and aligned loss weights."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from itertools import batched
from pathlib import Path

from apertus_common import Conversation, SystemPrompt, checkers_for, load_encoding


def records(path: Path | None) -> Iterator[Conversation]:
    if path is not None:
        with path.open(encoding="utf-8") as source:
            for line in source:
                yield Conversation.from_json(line)
        return
    yield (
        Conversation.builder()
        .system(SystemPrompt.build())
        .user("What is 6 times 7?")
        .think("Multiply the numbers.", loss_weight=0)
        .reply("42", loss_weight=0.5)
        .wait(loss_weight=1)
        .build()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    encoding = load_encoding(args.artifact)
    checks = checkers_for("full", allow_loss_weights=True)
    count = total_tokens = 0
    for batch in batched(records(args.input), args.batch_size, strict=False):
        for conversation in batch:
            report = conversation.check(checkers=checks)
            if not report.ok or not report.complete:
                raise ValueError(f"record did not pass full checking: {report}")
        encoded = encoding.encode_conversations(batch, return_loss_weights=True)
        for result in encoded:
            assert len(result.token_ids) == len(result.loss_weights)
            assert result.loss_weights[0] == 0
            count += 1
            total_tokens += len(result.token_ids)
    print(f"checked and encoded {count} conversations, {total_tokens} tokens")


if __name__ == "__main__":
    main()
