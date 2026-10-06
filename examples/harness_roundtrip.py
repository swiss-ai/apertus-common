"""Exact-token continuation mechanics, using supplied output rather than a server.

A native vLLM completions caller sends the entire retained prompt on every
request, enables token-ID responses, applies the special-ID mask and suspends
at output_end. Wait remains EOS. Message-end request suspension still needs
upstream specification clarification before production rollout.
"""

from __future__ import annotations

import argparse

from apertus_common import (
    CheckContext,
    Conversation,
    ParseBoundary,
    Reply,
    StreamParseError,
    SystemPrompt,
    Think,
    User,
    Wait,
    load_encoding,
)


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("artifact")
    directory: str = args.parse_args().artifact
    if not directory:
        args.error("supply an artifact directory")
    encoding = load_encoding(directory)
    history = Conversation(
        system=SystemPrompt.build(), items=[User(payload="Explain briefly.")]
    )
    prompt = encoding.encode_conversation(history)
    forbidden = frozenset(encoding.generation_constraints.forbidden_token_ids)
    request_settings = {
        "stop_token_ids": [encoding.controls.output_end],
        "return_token_ids": True,
        "allowed_token_ids": [
            token for token in range(encoding.vocab_size) if token not in forbidden
        ],
    }
    # The engine's EOS is encoding.generation_constraints.eos_token_id (wait).
    # Native completions bypass the chat renderer's automatic mask setup.
    assert encoding.controls.wait in request_settings["allowed_token_ids"]

    generated = encoding.encode_items([Think(payload="One step.")])
    parser = encoding.output_parser()
    parser.push_tokens(generated)
    output = parser.finish()
    report = output.check(context=CheckContext(prefix=history))
    assert report.ok and report.complete
    assert parser.boundary is ParseBoundary.MESSAGE_END

    # No queued input: send P + O unchanged, including the closing token.
    continuation = prompt + generated
    assert continuation == encoding.encode_conversation(
        Conversation(system=history.system, items=[*history.items, *output.items])
    )

    # Queued input: append its encoded envelopes to the same retained tokens.
    queued = [User(payload="Also give an example.")]
    with_input = continuation + encoding.encode_items(queued)
    assert with_input[: len(continuation)] == continuation

    # A logical wait always requires new input, even with pending tool calls.
    ended = encoding.parse_output(
        encoding.encode_items([Reply(payload="Done"), Wait()])
    )
    invalid_resume = Conversation(
        system=history.system,
        items=[*history.items, *ended.items, Think(payload="Unprompted")],
    ).check()
    assert not invalid_resume.ok

    # A malformed suffix is discarded at the parser's completed-message boundary.
    broken = [
        *generated,
        encoding.controls.output_start,
        *encoding.encode_text("call{"),
    ]
    recovery = encoding.output_parser()
    try:
        recovery.push_tokens(broken)
        recovery.finish()
    except StreamParseError as error:
        retained_output = broken[: error.last_valid_boundary]
        assert retained_output == generated
        assert error.completed == output
        safe_continuation = prompt + retained_output
        assert safe_continuation == continuation
    else:
        raise AssertionError("expected an incomplete header")
    print("Exact continuation, queued input, wait checks and suffix recovery passed.")


if __name__ == "__main__":
    main()
