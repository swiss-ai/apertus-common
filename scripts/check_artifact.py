"""Required integration check against a caller-supplied real tokenizer artifact.

Run with an artifact directory argument or APERTUS_ENCODING. Missing artifacts
fail explicitly; synthetic unit fixtures cannot substitute for this check.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, cast

from apertus_common import (
    Call,
    Conversation,
    Message,
    MessageCompleted,
    ParseFailure,
    Reply,
    StreamParseError,
    SystemPrompt,
    Think,
    TokenizerError,
    ToolSpec,
    UnknownOutput,
    User,
    Wait,
    load_encoding,
    render_str,
)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def check_artifact(directory: Path) -> tuple[int, int]:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer = cast(Any, AutoTokenizer).from_pretrained(
        directory, local_files_only=True
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    encoding = load_encoding(tokenizer)
    assert encoding.controls == load_encoding(directory).controls
    glyphs = "<|in|><|/in|><|out|><|/out|><|hdr|><|wait|><|pad|>"
    samples = [
        glyphs,
        " ".join(
            encoding.id_to_token(i) or "" for i in sorted(encoding.special_token_ids)
        ),
        "  e\u0301\r\n\x00\t🙂漢字\ufffd  ",
        "".join(chr(code) for code in range(256)),
        "".join(
            chr(code)
            for code in range(0, 0x110000, 421)
            if not 0xD800 <= code <= 0xDFFF
        ),
    ]
    for sample in samples:
        ids = encoding.encode_text(sample)
        require(
            encoding.special_token_ids.isdisjoint(ids),
            "ordinary text produced special IDs",
        )
    prompt = Conversation(
        system=SystemPrompt.build(
            tools=[
                ToolSpec(
                    name="echo", description="Echo", json_schema={"type": "object"}
                )
            ]
        ),
        items=[User(payload="Preserve this text: é")],
    )
    prompt_ids = encoding.encode_conversation(prompt)
    require(prompt_ids[0] == encoding.bos_token_id, "prompt does not start with BOS")
    require(
        encoding.decode_text(prompt_ids[1:]) == render_str(prompt),
        "system/prompt encoding changed wire text",
    )
    # NFC-stable payloads, so the tokenizer's normalizer leaves them unchanged.
    output = Conversation(
        items=[
            Think(payload="Inspect 🙂 and é."),
            Call(name="echo", counter=0, payload="malformed arguments retained"),
            UnknownOutput(
                output_type="extension", headers={"key": "value"}, payload="extra"
            ),
            Reply(payload=glyphs + "  é\r\n\x00\t🙂漢字\ufffd  "),
            Wait(),
        ]
    )
    ids = encoding.encode_items(output.items)
    require(
        encoding.parse_output(ids) == output, "batch token round trip changed messages"
    )
    for chunk_size in [1, 2, 7, 64, len(ids)]:
        parser = encoding.output_parser()
        completed: list[Message] = []
        for offset in range(0, len(ids), chunk_size):
            parser.push_tokens(ids[offset : offset + chunk_size])
            completed.extend(
                event.message
                for event in parser.take_events()
                if isinstance(event, MessageCompleted)
            )
        result = parser.finish()
        require(
            result == output,
            f"stream round trip changed messages at chunk size {chunk_size}",
        )
        require(
            completed == list(output.items[:-1]),
            "completion events differ from batch output",
        )
        require(parser.finish() is result, "finish is not idempotent")
        require(
            parser.last_valid_boundary == len(ids),
            "token recovery boundary is incorrect",
        )
    # Golden IDs are tied to the pinned production recipe, not library constants.
    require(
        encoding.encode_items([Reply(payload="Hello"), Wait()])
        == [10, 120213, 9, 36971, 11, 12],
        "pinned production golden IDs changed",
    )
    require(
        encoding.bos_token_id == 1 and encoding.id_to_token(1) == "<s>",
        "pinned production BOS changed",
    )
    for word in (
        "system",
        "host",
        "user",
        "document",
        "result",
        "think",
        "reply",
        "claim",
        "call",
        "name",
        "counter",
    ):
        word_ids = encoding.encode_text(word)
        require(len(word_ids) == 1, f"profile word is not one token: {word}")
        if word not in ("name", "counter"):
            require(
                encoding.encode_text(word + '{"name":"echo","counter":0}')[0]
                == word_ids[0],
                f"header context changed type token: {word}",
            )
    controls = encoding.controls
    allowed = {
        controls.output_start,
        controls.output_end,
        controls.header_end,
        controls.wait,
    }
    require(
        frozenset(encoding.generation_constraints.forbidden_token_ids)
        == encoding.special_token_ids - allowed,
        "generation mask differs from artifact special inventory",
    )
    require(
        encoding.generation_constraints.eos_token_id == controls.wait,
        "EOS must be wait",
    )
    try:
        encoding.parse_output([controls.wait, controls.output_start])
    except StreamParseError as error:
        require(
            error.reason is ParseFailure.TRAILING_AFTER_WAIT,
            "wrong error for trailing tokens",
        )
    else:
        raise AssertionError("trailing input after wait was accepted")
    return encoding.vocab_size, len(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact", nargs="?", default=os.environ.get("APERTUS_ENCODING")
    )
    args = parser.parse_args()
    artifact = cast(str | None, args.artifact)
    if not artifact:
        parser.error(
            "supply an artifact directory or set APERTUS_ENCODING; this check never skips"
        )
    directory = Path(artifact)
    try:
        vocab_size, output_tokens = check_artifact(directory)
    except (AssertionError, TokenizerError, StreamParseError) as error:
        print(f"artifact check FAILED: {error}")
        return 1
    print(
        f"artifact check OK: {directory.resolve()} ({vocab_size} vocabulary IDs; {output_tokens} output tokens; batch and five stream chunk sizes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
