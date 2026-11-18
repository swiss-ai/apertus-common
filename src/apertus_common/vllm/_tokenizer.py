"""vLLM tokenizer for the ``apertus2`` mode.

Follows ``vllm.tokenizers.deepseek_v4``: the HF fast tokenizer of the
checkpoint is wrapped so that ``apply_chat_template`` renders the Apertus 2
format with apertus-common instead of a Jinja template. Every other tokenizer
operation (``encode``, ``decode``, vocab) is the unchanged HF tokenizer, so
``/v1/completions`` keeps working on raw text.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from transformers import TokenizersBackend
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm.tokenizers.protocol import TokenizerLike

from apertus_common.core import render_str
from apertus_common.encoding import ApertusEncoding, load_encoding
from apertus_common.formats import from_openai

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

# OpenAI reasoning_effort values outside the format's low/medium/high.
_EFFORT_ALIASES = {"none": "low", "minimal": "low", "max": "high", "xhigh": "high"}

# Includes reasoning_content: from_openai keeps prior thinking as think frames.
_HF_MESSAGE_KEYS = {
    "role",
    "content",
    "name",
    "tool_calls",
    "tool_call_id",
    "reasoning_content",
}


def _clean_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the OpenAI keys apertus-common maps; drop vLLM-only extras."""
    cleaned = []
    for message in messages:
        kept = {
            k: v for k, v in message.items() if k in _HF_MESSAGE_KEYS and v is not None
        }
        calls = kept.get("tool_calls")
        if calls:
            kept["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"].get("arguments") or "{}",
                    },
                }
                for call in calls
            ]
        cleaned.append(kept)
    return cleaned


def encode_chat(
    encoding: ApertusEncoding,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    reasoning_effort: str | None = None,
    system_defaults: Mapping[str, Any] | None = None,
    tokenize: bool = True,
) -> list[int] | str:
    """Map OpenAI messages to an Apertus 2 conversation ending at ``<|/in|>``."""
    if reasoning_effort is not None:
        reasoning_effort = _EFFORT_ALIASES.get(reasoning_effort, reasoning_effort)
    result = from_openai(
        _clean_messages(messages),
        tools=list(tools) if tools else None,
        reasoning_effort=reasoning_effort,
        system_defaults=dict(system_defaults) if system_defaults else None,
    )
    for diagnostic in result.diagnostics:
        if diagnostic.lossy:
            logger.warning("apertus2 chat mapping: %s", diagnostic.message)
    if not tokenize:
        return render_str(result.value)
    return encoding.encode_conversation(result.value)


def _wrap(tokenizer: TokenizersBackend, encoding: ApertusEncoding) -> TokenizersBackend:
    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()

    class _Apertus2Tokenizer(tokenizer.__class__):  # type: ignore[misc,name-defined]
        apertus_encoding = encoding

        def apply_chat_template(  # type: ignore[override]
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str | list[int]:
            # vLLM's renderer passes the parsed conversation (content parts
            # flattened to strings) next to the raw request messages.
            conversation = kwargs.get("conversation", messages)
            effort = kwargs.get("reasoning_effort")
            if not isinstance(effort, str):
                effort = None
            return encode_chat(
                encoding,
                conversation,
                tools=tools,
                reasoning_effort=effort,
                system_defaults=kwargs.get("system_defaults"),
                tokenize=kwargs.get("tokenize", True),
            )

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return _rebuild, (tokenizer, encoding)

    _Apertus2Tokenizer.__name__ = f"Apertus2{tokenizer.__class__.__name__}"
    wrapped.__class__ = _Apertus2Tokenizer
    return wrapped


def _rebuild(
    tokenizer: TokenizersBackend, encoding: ApertusEncoding
) -> TokenizersBackend:
    return _wrap(tokenizer, encoding)


class Apertus2Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> TokenizersBackend:
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        # apertus-common binds its own copy: load_encoding changes how the
        # tokenizer it is given treats special tokens, and vLLM's copy must
        # keep encoding raw text (e.g. /v1/completions) unchanged.
        encoding = load_encoding(TokenizersBackend.from_pretrained(*args, **kwargs))
        return get_cached_tokenizer(_wrap(tokenizer, encoding))
