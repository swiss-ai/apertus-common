"""vLLM renderer for the ``apertus2`` tokenizer mode.

Mirrors ``vllm.renderers.deepseek_v4``: chat messages are parsed by vLLM
(content parts flattened to strings), then handed to the tokenizer wrapper's
``apply_chat_template`` which returns token IDs from apertus-common.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.entrypoints.chat_utils import (
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.base import BaseRenderer
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.utils.async_utils import make_async

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.entrypoints.chat_utils import (
        ChatCompletionMessageParam,
        ConversationMessage,
    )
    from vllm.renderers.inputs import DictPrompt
    from vllm.renderers.params import ChatParams
    from vllm.tokenizers import TokenizerLike


class Apertus2Renderer(BaseRenderer["TokenizerLike"]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: TokenizerLike | None,
    ) -> None:
        super().__init__(config, tokenizer)
        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(self, *args: Any, **kwargs: Any) -> list[int] | str:
        return self.get_tokenizer().apply_chat_template(*args, **kwargs)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
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
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        prompt_raw = await self._apply_chat_template_async(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )
        return conversation, self._finish(prompt_raw, mm_data, mm_uuids)

    @staticmethod
    def _finish(prompt_raw: list[int] | str, mm_data: Any, mm_uuids: Any) -> DictPrompt:
        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids
        return prompt
