"""Report parse throughput and peak memory, with no hardware-specific gate.

Run ``uv run python scripts/bench_parse.py``; add ``--artifact PATH`` for the
verified tokenizer artifact. Cases cover long payloads, many messages and small
chunks. Event queues are drained as a streaming consumer would drain them.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections.abc import Callable

from apertus_common import Conversation, Reply, TextStreamParser, Wait, render_str


def measure(label: str, size: int, action: Callable[[], object], repeats: int) -> None:
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - started)
    tracemalloc.start()
    action()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"{label:32s} {size / best / 1e6:8.2f} Munits/s  {best * 1000:8.1f} ms  peak {peak / 1e6:.2f} MB"
    )


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--artifact")
    args.add_argument("--repeats", type=int, default=3)
    options = args.parse_args()
    fixtures = {
        "long": Conversation(items=(Reply(payload="e\u0301雪🚀 " * 100_000), Wait())),
        "many": Conversation(
            items=(*(Reply(payload="short reply") for _ in range(3_000)), Wait())
        ),
    }
    for name, conversation in fixtures.items():
        text = render_str(conversation)
        for chunk_size in (len(text), 7, 128):

            def parse_text(text: str = text, chunk_size: int = chunk_size) -> object:
                parser = TextStreamParser()
                for offset in range(0, len(text), chunk_size):
                    parser.push_text(text[offset : offset + chunk_size])
                    parser.take_events()
                return parser.finish()

            measure(f"text/{name}/{chunk_size}", len(text), parse_text, options.repeats)
        if options.artifact:
            from apertus_common import ApertusEncoding, load_encoding

            encoding = load_encoding(options.artifact)
            tokens = encoding.encode_conversation(conversation)
            for chunk_size in (len(tokens), 1, 32):

                def parse_tokens(
                    tokens: list[int] = tokens,
                    chunk_size: int = chunk_size,
                    encoding: ApertusEncoding = encoding,
                ) -> object:
                    parser = encoding.output_parser()
                    for offset in range(0, len(tokens), chunk_size):
                        parser.push_tokens(tokens[offset : offset + chunk_size])
                        parser.take_events()
                    return parser.finish()

                measure(
                    f"tokens/{name}/{chunk_size}",
                    len(tokens),
                    parse_tokens,
                    options.repeats,
                )


if __name__ == "__main__":
    main()
