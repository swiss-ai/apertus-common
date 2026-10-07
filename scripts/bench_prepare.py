"""Measure loading, reusable checking and scalar/batch encoding on bounded data."""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections.abc import Callable
from functools import partial

from apertus_common import (
    ArgumentsChecker,
    Call,
    Checker,
    Conversation,
    Host,
    Reply,
    Result,
    SystemPrompt,
    ToolSpec,
    User,
    Wait,
    checkers_for,
    load_encoding,
)


def measure(
    label: str, count: int, action: Callable[[], object], *, unit: str = "records"
) -> None:
    # Warm imports/caches; time independently of tracemalloc instrumentation.
    action()
    start = time.perf_counter()
    action()
    duration = time.perf_counter() - start
    tracemalloc.start()
    action()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"{label:30} {count / duration:10.1f} {unit}/s {peak / 1e6:8.2f} MB Python peak"
    )


def fixtures(count: int, *, varied: bool, long: bool) -> list[Conversation]:
    return [
        Conversation(
            system=SystemPrompt.build(
                tools=[
                    ToolSpec(
                        name="lookup",
                        description="Lookup a number",
                        json_schema={
                            "type": "object",
                            "properties": {"number": {"type": "integer"}},
                            "required": ["number"],
                            "title": str(index if varied else 0),
                        },
                    )
                ]
            ),
            items=[
                User(payload="Look up a number"),
                Call.build(name="lookup", counter=0, arguments={"number": index}),
                Result(name="lookup", counter=0, payload="done"),
                Reply(
                    payload="e\u0301雪🚀 <|out|> " * (1024 if long else 1),
                    loss_weight=0.5,
                ),
                Wait(),
            ],
        )
        for index in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--records", type=int, default=128)
    options = parser.parse_args()
    if options.records < 1:
        parser.error("--records must be positive")
    encoding = load_encoding(options.artifact)

    def encode_scalar(batch: list[Conversation]) -> list[list[int]]:
        return [encoding.encode_conversation(c) for c in batch]

    for name, varied, long in [
        ("repeated", False, False),
        ("varied", True, False),
        ("long", False, True),
    ]:
        batch = fixtures(options.records, varied=varied, long=long)
        native = [conversation.to_json() for conversation in batch]
        checks = checkers_for("full", allow_loss_weights=True)
        uncached = (*checks[:-1], ArgumentsChecker(schema_cache_size=0))

        def load(native: list[str] = native) -> list[Conversation]:
            return [Conversation.from_json(record) for record in native]

        def check(
            batch: list[Conversation] = batch, checks: tuple[Checker, ...] = checks
        ) -> None:
            for conversation in batch:
                report = conversation.check(checkers=checks)
                assert report.ok and report.complete

        def total() -> object:
            loaded = load()
            check(loaded)
            return encoding.encode_conversations(loaded, return_loss_weights=True)

        measure(f"{name}/load", len(batch), load)
        measure(f"{name}/check-warm", len(batch), check)
        measure(f"{name}/check-no-cache", len(batch), partial(check, batch, uncached))
        measure(f"{name}/scalar", len(batch), partial(encode_scalar, batch))
        measure(
            f"{name}/batch", len(batch), partial(encoding.encode_conversations, batch)
        )
        measure(
            f"{name}/weighted",
            len(batch),
            partial(encoding.encode_conversations, batch, return_loss_weights=True),
        )
        measure(f"{name}/total-warm", len(batch), total)

    for size in (100, 1000, 10000):
        items: list[Call | Result | Host] = []
        for counter in range(size):
            items.extend(
                [
                    Call(name="lookup", counter=counter, payload="{}"),
                    Result(name="lookup", counter=counter, payload="done"),
                    Host(payload="notice"),
                ]
            )
        history = Conversation(
            system=SystemPrompt.build(
                tools=[
                    ToolSpec(
                        name="lookup",
                        description="Lookup",
                        json_schema={"type": "object"},
                    )
                ]
            ),
            items=items,
        )
        measure(
            f"closed-host/{size}",
            size,
            partial(history.check, "conversation"),
            unit="call/result/host groups",
        )


if __name__ == "__main__":
    main()
