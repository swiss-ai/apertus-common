"""Measure parsing throughput to record implementation baselines.

Builds a synthetic rendered output-only transcript, then reports scanning
throughput for the text lexer alone and end-to-end throughput for
``StreamParser.parse``. Baselines feed a future regression budget; the
script sets no budget itself.

Run: ``uv run python scripts/bench_parse.py``
"""

from __future__ import annotations

import time
from collections.abc import Callable

from apertus_common import (
    Assistant,
    Conversation,
    OutputMessage,
    Think,
    ToolCall,
    Wait,
    render_str,
)
from apertus_common.core._scan import TextLexer
from apertus_common.parsing import StreamParser

MESSAGES = 3_000
REPEATS = 5


def build_transcript(messages: int) -> str:
    """Render an output-only conversation with realistic mixed messages."""
    items: list[OutputMessage | Wait] = []
    for index in range(messages):
        cycle = index % 3
        if cycle == 0:
            items.append(Think(payload="considering the request " * 20))
        elif cycle == 1:
            items.append(
                ToolCall(
                    id=f"search:{index}",
                    name="search",
                    args={"query": "benchmark " * 10, "limit": index},
                )
            )
        else:
            items.append(Assistant(payload="a helpful streamed reply " * 40))
    items.append(Wait())
    return render_str(Conversation.model_validate({"items": items}))


def best_seconds(action: Callable[[], None], repeats: int) -> float:
    """Return the fastest wall-clock time of *repeats* runs of *action*."""
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - started)
    return best


def main() -> None:
    text = build_transcript(MESSAGES)
    megabytes = len(text.encode()) / 1_000_000

    def scan() -> None:
        lexer = TextLexer()
        lexer.push(text)
        lexer.finish()

    def parse() -> None:
        StreamParser.parse(text)

    scan_seconds = best_seconds(scan, REPEATS)
    parse_seconds = best_seconds(parse, REPEATS)

    print(f"transcript: {MESSAGES} messages, {megabytes:.1f} MB")
    print(
        f"TextLexer scan:     {megabytes / scan_seconds:8.1f} MB/s"
        f"  ({scan_seconds * 1000:.1f} ms)"
    )
    print(
        f"StreamParser.parse: {megabytes / parse_seconds:8.1f} MB/s"
        f"  ({parse_seconds * 1000:.1f} ms,"
        f" {MESSAGES / parse_seconds:,.0f} messages/s)"
    )


if __name__ == "__main__":
    main()
