"""vLLM renderer for the ``apertus2`` tokenizer mode.

Mirrors ``vllm.renderers.deepseek_v4``: chat messages are parsed by vLLM
(content parts flattened to strings), then handed to the tokenizer wrapper's
``apply_chat_template``, which returns token IDs from apertus-common. The
prompt is always token IDs: text would be re-tokenized by vLLM, which turns
control glyphs written in message text into control tokens.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.chat_utils import (
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.base import BaseRenderer
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.utils.async_utils import make_async

from apertus_common.vllm import MODE
from apertus_common.vllm._parser import FLAGS

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ConversationMessage,
    )
    from vllm.renderers.inputs import DictPrompt
    from vllm.renderers.params import ChatParams
    from vllm.tokenizers import TokenizerLike


def _calls(messages: Any) -> list[dict[str, Any]]:
    """Function dicts of assistant tool calls, in order."""
    functions = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict):
                functions.append(function)
    return functions


def _hide_invalid_arguments(messages: Any) -> tuple[Any, dict[int, str]]:
    """Copy messages with non-object tool-call arguments replaced by ``{}``.

    vLLM parses replayed arguments as JSON and fails on invalid ones, but the
    profile answers invalid arguments with a tool error, so the call must
    replay as generated. Returns the copy and the raw strings by call position.
    """
    invalid = {}
    for position, function in enumerate(_calls(messages)):
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            is_object = isinstance(json.loads(arguments), dict)
        except ValueError:
            is_object = False
        # vLLM would also turn "" and non-object JSON such as null into {}.
        if not is_object:
            invalid[position] = arguments
    if not invalid:
        return messages, invalid
    parsable = copy.deepcopy(messages)
    functions = _calls(parsable)
    for position in invalid:
        functions[position]["arguments"] = "{}"
    return parsable, invalid


def _restore_invalid_arguments(conversation: Any, invalid: dict[int, str]) -> None:
    if not invalid:
        return
    functions = _calls(conversation)
    if max(invalid) >= len(functions):
        raise ValueError("Could not align replayed tool calls")
    for position, arguments in invalid.items():
        functions[position]["arguments"] = arguments


class Apertus2Renderer(BaseRenderer["TokenizerLike"]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: TokenizerLike | None,
    ) -> None:
        if getattr(config, "speculative_config", None) is not None:
            raise ValueError(
                "Apertus 2 masks forbidden tokens with logit_bias, which vLLM "
                "ignores under speculative decoding; serve without it"
            )
        # Without the reasoning parser, vLLM skips the request guards and the
        # mask for requests without tools.
        outputs = getattr(config, "structured_outputs_config", None)
        if getattr(outputs, "reasoning_parser", None) != MODE:
            raise ValueError(FLAGS)
        super().__init__(config, tokenizer)
        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(self, *args: Any, **kwargs: Any) -> list[int]:
        kwargs["tokenize"] = True
        return self.get_tokenizer().apply_chat_template(*args, **kwargs)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        parsable, invalid = _hide_invalid_arguments(messages)
        conversation, mm_data, mm_uuids = parse_chat_messages(
            parsable,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        _restore_invalid_arguments(conversation, invalid)
        prompt_raw = self._apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )
        return conversation, self._finish(prompt_raw, mm_data, mm_uuids)

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        parsable, invalid = _hide_invalid_arguments(messages)
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            parsable,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        _restore_invalid_arguments(conversation, invalid)
        prompt_raw = await self._apply_chat_template_async(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )
        return conversation, self._finish(prompt_raw, mm_data, mm_uuids)

    @staticmethod
    def _finish(prompt_raw: list[int], mm_data: Any, mm_uuids: Any) -> DictPrompt:
        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids
        return prompt
