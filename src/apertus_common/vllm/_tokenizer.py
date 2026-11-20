"""vLLM tokenizer for the ``apertus2`` mode.

Follows ``vllm.tokenizers.deepseek_v4``: the HF fast tokenizer of the
checkpoint is wrapped so that ``apply_chat_template`` renders the Apertus 2
format with apertus-common instead of a Jinja template. Every other tokenizer
operation (``encode``, ``decode``, vocab) is the unchanged HF tokenizer, so
``/v1/completions`` keeps working on raw text.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from transformers import TokenizersBackend
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm.tokenizers.protocol import TokenizerLike

from apertus_common.core import render_str
from apertus_common.encoding import ApertusEncoding, load_encoding
from apertus_common.errors import ApertusFormatError
from apertus_common.formats import (
    MappingPolicy,
    OpenAIMappingContext,
    ToolAssociation,
    from_openai,
)

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

# OpenAI reasoning_effort values above the format's low/medium/high.
_EFFORT_ALIASES = {"max": "high", "xhigh": "high"}
_THINKING_OFF = {"none", "minimal"}

# Includes reasoning_content: from_openai keeps prior thinking as think frames.
# Participant names have no Apertus meaning and are dropped.
_HF_MESSAGE_KEYS = {
    "role",
    "content",
    "tool_calls",
    "tool_call_id",
    "reasoning_content",
}

# Leading system and developer messages merge into the behavior field.
_POLICY = MappingPolicy(merge_system_messages=True)

# Tool-call IDs minted by the parser; they carry the call counter.
CALL_ID = re.compile(r"apertus_(\d+)")


def call_id(counter: int) -> str:
    return f"apertus_{counter}"


def _context(messages: Sequence[Mapping[str, Any]]) -> OpenAIMappingContext:
    """Counters of replayed calls whose IDs this integration minted.

    A client that drops earlier messages keeps the counters of the calls it
    replays; other IDs are numbered after them in order.
    """
    associations: dict[int, ToolAssociation] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            match = CALL_ID.fullmatch(str(call.get("id")))
            name = call["function"]["name"]
            if match and isinstance(name, str) and name:
                counter = int(match[1])
                associations[counter] = ToolAssociation(
                    id=match[0], name=name, counter=counter
                )
    return OpenAIMappingContext(
        associations=tuple(associations.values()),
        next_counter=max(associations, default=-1) + 1,
    )


def _arguments_json(arguments: Any) -> str:
    """A tool call's arguments as the JSON string ``from_openai`` takes.

    vLLM parses the request's JSON string into a dict before the tokenizer sees the
    conversation, so a call with arguments arrives as a dict (``{}`` for none).
    Dicts are serialized as compact JSON, the form the model emits, so canonical
    arguments replay to the generated tokens. vLLM has already discarded the
    original string, so other spellings (escapes, number formats, whitespace)
    are normalized. Strings pass unchanged: the renderer keeps arguments that
    are not a JSON object as generated.
    """
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


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
                        "arguments": _arguments_json(call["function"].get("arguments")),
                    },
                }
                for call in calls
            ]
        # vLLM turns a missing content into "", which would encode an empty
        # reply the model never generated.
        if (
            kept.get("role") == "assistant"
            and kept.get("content") == ""
            and (calls or "reasoning_content" in kept)
        ):
            del kept["content"]
        cleaned.append(kept)
    return cleaned


def encode_chat(
    encoding: ApertusEncoding,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    reasoning_effort: str | None = None,
    system_defaults: Mapping[str, Any] | None = None,
    enable_thinking: bool | None = None,
    tokenize: bool = True,
) -> list[int] | str:
    """Map OpenAI messages to an Apertus 2 conversation ending at ``<|/in|>``.

    Invalid input raises ``ValueError``, which vLLM returns as HTTP 400.
    """
    if messages and messages[-1].get("role") == "assistant":
        raise ValueError(
            "Apertus 2 generates after an input message: end the conversation "
            "with a user or tool message (assistant prefill is unsupported)"
        )
    if reasoning_effort in _THINKING_OFF or enable_thinking is False:
        raise ValueError(
            "Apertus 2 always thinks: use reasoning_effort low, medium or high"
        )
    if reasoning_effort is not None:
        reasoning_effort = _EFFORT_ALIASES.get(reasoning_effort, reasoning_effort)
    try:
        cleaned = _clean_messages(messages)
        result = from_openai(
            cleaned,
            tools=list(tools) if tools else None,
            reasoning_effort=reasoning_effort,
            system_defaults=dict(system_defaults) if system_defaults else None,
            context=_context(cleaned),
            policy=_POLICY,
        )
        for diagnostic in result.diagnostics:
            # Merging system and developer messages is the serving policy.
            if diagnostic.lossy and diagnostic.code != "system-merged":
                logger.warning("apertus2 chat mapping: %s", diagnostic.message)
        if not tokenize:
            return render_str(result.value)
        return encoding.encode_conversation(result.value)
    except ApertusFormatError as error:
        raise ValueError(f"Invalid Apertus 2 conversation: {error}") from error
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(f"Malformed chat message: {error!r}") from error


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
                enable_thinking=kwargs.get("enable_thinking"),
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
