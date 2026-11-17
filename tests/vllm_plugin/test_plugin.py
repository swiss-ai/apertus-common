"""Tests for the vLLM integration; skipped unless vLLM is importable.

Run them in a CPU-only vLLM environment, e.g. the Docker image described in
README.md, with ``APERTUS_ENCODING`` pointing at an ``Apertus_2_instruct``
tokenizer directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

vllm = pytest.importorskip("vllm")

from apertus_common import from_openai, load_encoding
from apertus_common.vllm import register
from apertus_common.vllm._parser import Apertus2Parser
from apertus_common.vllm._tokenizer import Apertus2Tokenizer

ARTIFACT = os.environ.get("APERTUS_ENCODING")
pytestmark = pytest.mark.skipif(not ARTIFACT, reason="APERTUS_ENCODING not set")

MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Who is Pablo Picasso?"},
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather by city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


@pytest.fixture(scope="module")
def tokenizer():
    return Apertus2Tokenizer.from_pretrained(Path(ARTIFACT))


@pytest.mark.parametrize("arguments", [{"city": "SF"}, {}, "{}", None])
def test_tool_call_history_accepts_arguments_as_vllm_parses_them(
    tokenizer, arguments
) -> None:
    # vLLM turns a call's JSON arguments into a dict before the tokenizer sees them.
    messages = [
        {"role": "user", "content": "Weather in SF?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False)
    expected = (
        json.dumps(arguments, ensure_ascii=False)
        if isinstance(arguments, dict)
        else (arguments or "{}")
    )
    assert expected in prompt


def test_register_populates_vllm_registries() -> None:
    from vllm.reasoning import ReasoningParserManager
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm.tool_parsers import ToolParserManager

    register()
    register()  # idempotent: vLLM loads plugins in several processes
    assert TokenizerRegistry.load_tokenizer_cls("apertus2") is Apertus2Tokenizer
    assert (
        RENDERER_REGISTRY.load_renderer_cls("apertus2").__name__ == "Apertus2Renderer"
    )
    assert ReasoningParserManager.get_reasoning_parser("apertus2").__name__.endswith(
        "ReasoningAdapter"
    )
    assert ToolParserManager.get_tool_parser("apertus2").__name__.endswith(
        "ToolAdapter"
    )


def test_apply_chat_template_matches_library_encoding(tokenizer) -> None:
    ids = tokenizer.apply_chat_template(
        MESSAGES, conversation=MESSAGES, chat_template=None, return_dict=False
    )
    expected = load_encoding(Path(ARTIFACT)).encode_conversation(
        from_openai(MESSAGES).value
    )
    assert ids == expected
    rendered = tokenizer.apply_chat_template(
        MESSAGES, tools=TOOLS, reasoning_effort="low", tokenize=False
    )
    assert rendered.endswith("<|in|>user<|hdr|>Who is Pablo Picasso?<|/in|>")
    assert '"thinking":"low"' in rendered
    assert "get_weather" in rendered


def test_plain_text_encoding_is_untouched(tokenizer) -> None:
    # /v1/completions must still see control glyphs in text as special tokens.
    encoding = load_encoding(Path(ARTIFACT))
    assert tokenizer.encode("<|in|>", add_special_tokens=False) == [
        encoding.controls.input_start
    ]


THINK_REPLY = (
    "<|out|>think<|hdr|>Let me think.<|/out|><|out|>reply<|hdr|>A painter.<|/out|>"
)
CALL = (
    "<|out|>think<|hdr|>Need weather.<|/out|>"
    '<|out|>call{"name":"get_weather","counter":-1}<|hdr|>{"city":"SF"}<|/out|>'
)
TWO_CALLS = (
    "<|out|>reply<|hdr|>Sure.<|/out|>"
    '<|out|>call{"name":"get_weather","counter":-1}<|hdr|>{"city":"SF"}<|/out|>'
    '<|out|>call{"name":"get_weather","counter":-2}<|hdr|>{"city":"NYC"}<|/out|>'
)


@pytest.mark.parametrize(
    ("text", "reasoning", "content", "calls"),
    [
        (THINK_REPLY, "Let me think.", "A painter.", []),
        ("<|out|>think<|hdr|><|/out|><|out|>reply<|hdr|>Hi!<|/out|>", None, "Hi!", []),
        ("<|out|>reply<|hdr|>Hi!<|/out|><|wait|>", None, "Hi!", []),
        (CALL, "Need weather.", None, [("get_weather", {"city": "SF"})]),
        (
            TWO_CALLS,
            None,
            "Sure.",
            [("get_weather", {"city": "SF"}), ("get_weather", {"city": "NYC"})],
        ),
        ("<|out|>claim<|hdr|>42<|/out|>", None, "42", []),
    ],
)
def test_single_pass_parse(tokenizer, text, reasoning, content, calls) -> None:
    import json

    parser = Apertus2Parser(tokenizer)
    got_reasoning, got_content, info = parser._single_pass_parse(text, [])
    assert got_reasoning == reasoning
    assert got_content == content
    assert [
        (c.function.name, json.loads(c.function.arguments)) for c in info.tool_calls
    ] == calls


@pytest.mark.parametrize("text", [THINK_REPLY, CALL, TWO_CALLS])
@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_streaming_matches_single_pass(tokenizer, text, chunk) -> None:
    import json

    parser = Apertus2Parser(tokenizer)
    reasoning, content, info = parser._single_pass_parse(text, [])

    parser._reset()
    got = {"reasoning": "", "content": "", "tools": {}}
    deltas = [
        parser._events_to_delta(parser._feed(text[i : i + chunk], []))
        for i in range(0, len(text), chunk)
    ]
    deltas.append(parser.finish_streaming())
    for delta in deltas:
        if delta is None:
            continue
        got["reasoning"] += delta.reasoning or ""
        got["content"] += delta.content or ""
        for call in delta.tool_calls or []:
            slot = got["tools"].setdefault(call.index, {"name": "", "args": ""})
            if call.function and call.function.name:
                slot["name"] = call.function.name
            if call.function and call.function.arguments:
                slot["args"] += call.function.arguments
    assert got["reasoning"].rstrip() == (reasoning or "")
    assert got["content"].strip() == (content or "")
    assert [(s["name"], json.loads(s["args"])) for s in got["tools"].values()] == [
        (c.function.name, json.loads(c.function.arguments)) for c in info.tool_calls
    ]


def test_reasoning_state_from_token_ids(tokenizer) -> None:
    parser = Apertus2Parser(tokenizer)
    ids = tokenizer.encode(THINK_REPLY, add_special_tokens=False)
    hdr = ids.index(tokenizer.encode("<|hdr|>", add_special_tokens=False)[0])
    assert not parser.is_reasoning_end(ids[: hdr + 2])
    assert parser.is_reasoning_end(ids)
    think_tokens = len(tokenizer.encode("Let me think.", add_special_tokens=False))
    assert parser.count_reasoning_tokens(ids) == think_tokens
