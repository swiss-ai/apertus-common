"""vLLM parser for Apertus 2 output messages.

Generated output is a burst of output messages ending at ``<|wait|>``::

    <|out|>think<|hdr|>...reasoning...<|/out|>
    <|out|>reply<|hdr|>...visible text...<|/out|>
    <|out|>call{"name":"get_weather","counter":0}<|hdr|>{"city":"SF"}<|/out|>
    <|wait|>

The parser reads the generated token IDs with apertus-common's
``TokenStreamParser``, so only control token IDs frame messages; text that
spells a control glyph stays text. Think messages become ``reasoning``, reply
messages ``content`` and call messages ``tool_calls``. Streaming and
non-streaming responses share one projection. Within one streamed delta,
reasoning and content are separate fields, so their order inside a delta that
spans several messages is not kept.

vLLM's serving layer calls :meth:`Apertus2Parser.parse` and
:meth:`Apertus2Parser.parse_delta` for chat completions and responses when
both parser flags name ``apertus2`` and auto tool choice is on. The class is a
:class:`ParserEngine` only so that vLLM selects it as that one parser; its
declarative text config is empty and never frames output. The adapters vLLM
uses for any other flag combination refuse to parse.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.parser.engine.adapters import (
    ParserEngineReasoningAdapter,
    ParserEngineToolAdapter,
)
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import ParserEngineConfig

from apertus_common.errors import ParseFailure, StreamParseError, TokenizerError
from apertus_common.model import Call
from apertus_common.parsing import (
    MessageCompleted,
    MessageStart,
    ParseEvent,
    PayloadDelta,
    WaitReached,
)
from apertus_common.vllm._tokenizer import call_id

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

    from apertus_common.encoding import ApertusEncoding

logger = logging.getLogger(__name__)

# vLLM requires a declarative config; framing comes from token IDs instead.
_CONFIG = ParserEngineConfig(name="apertus2")

_FLAGS = (
    "Apertus 2 output needs --reasoning-parser apertus2, --tool-call-parser "
    "apertus2 and --enable-auto-tool-choice together"
)


@dataclass
class _Delta:
    reasoning: str = ""
    content: str = ""
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def extend(self, other: _Delta) -> None:
        self.reasoning += other.reasoning
        self.content += other.content
        self.calls += other.calls


class _Projection:
    """Project one generated token stream onto the chat response fields.

    Think and reply payloads stream as they arrive. A call is released only
    once its message completes, so a partial call is never returned. Claims
    are ignored, as the profile ignores a known type the deployment does not
    consume. Unknown output types are not shown as a reply. Output that is
    malformed or follows the wait is not projected; text already projected
    stays.
    """

    def __init__(self, encoding: ApertusEncoding, used_ids: set[str]) -> None:
        self._parser = encoding.output_parser()
        self._closed = False
        self._active: str | None = None
        self._call_ids = used_ids

    def push(self, token_ids: Sequence[int]) -> _Delta:
        delta = _Delta()
        if not token_ids or self._closed:
            return delta
        try:
            self._parser.push_tokens(token_ids)
        except StreamParseError as error:
            self._closed = True
            # Tokens after the wait only appear when a request disables it.
            trailing = error.reason is ParseFailure.TRAILING_AFTER_WAIT
            (logger.debug if trailing else logger.warning)(
                "Apertus 2: malformed output (%s at token %d); the rest is dropped",
                error.reason.value,
                error.offset,
            )
        self._project(self._parser.take_events(), delta)
        return delta

    def finish(self) -> _Delta:
        delta = _Delta()
        if self._closed:
            return delta
        self._closed = True
        try:
            self._parser.finish()
        except StreamParseError as error:
            # Usually max_tokens, which finish_reason already reports.
            logger.debug("Apertus 2: output ended early (%s)", error.reason.value)
        self._project(self._parser.take_events(), delta)
        return delta

    def _project(self, events: list[ParseEvent], delta: _Delta) -> None:
        for event in events:
            if isinstance(event, WaitReached):
                self._closed = True
            elif isinstance(event, MessageStart):
                self._active = event.type
                if event.type == "claim":
                    logger.debug("Apertus 2: claim %d ignored", event.index)
                elif event.type not in {"think", "reply", "call"}:
                    logger.warning(
                        "Apertus 2: unsupported output type %r omitted", event.type
                    )
            elif isinstance(event, PayloadDelta):
                if self._active == "think":
                    delta.reasoning += event.text
                elif self._active == "reply":
                    delta.content += event.text
            elif isinstance(event, MessageCompleted):
                self._active = None
                message = event.message
                if isinstance(message, Call):
                    delta.calls.append(
                        (self._call_id(message.counter), message.name, message.payload)
                    )

    def _call_id(self, counter: int) -> str:
        # A counter reused in this response or the history gets a distinct ID,
        # which replay numbers afresh.
        candidate = call_id(counter) if counter >= 0 else make_tool_call_id()
        while candidate in self._call_ids:
            candidate += "_"
        self._call_ids.add(candidate)
        return candidate


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _history_call_ids(request: Any) -> set[str]:
    """Tool-call IDs the request replays (chat messages, Responses items)."""
    ids = set()
    for message in getattr(request, "messages", None) or ():
        for call in _field(message, "tool_calls") or ():
            ids.add(str(_field(call, "id")))
    items = getattr(request, "input", None)
    for item in items if isinstance(items, list) else ():
        if _field(item, "type") == "function_call":
            ids.add(str(_field(item, "call_id")))
    return ids


def _tool_output_enabled(request: Any) -> bool:
    return bool(getattr(request, "tools", None)) and (
        getattr(request, "tool_choice", None) != "none"
    )


def _without_disabled_calls(delta: _Delta, request: Any) -> None:
    # Stock serving drops calls here when not streaming; drop them for both.
    if delta.calls and not _tool_output_enabled(request):
        logger.warning(
            "Apertus 2: %d tool call(s) dropped; the request disables tool output",
            len(delta.calls),
        )
        delta.calls = []


def _special_ids_in_bias(logit_bias: Any, special: frozenset[int]) -> bool:
    for key in logit_bias or {}:
        try:
            if int(key) in special:
                return True
        except (TypeError, ValueError):
            continue
    return False


class Apertus2Parser(ParserEngine):
    CONFIG_NAME = "apertus2"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("parser_engine_config", _CONFIG)
        super().__init__(tokenizer, tools, **kwargs)
        encoding = getattr(tokenizer, "apertus_encoding", None)
        if encoding is None:
            raise ValueError(
                "The apertus2 parser needs --tokenizer-mode apertus2, which "
                "provides the Apertus 2 encoding"
            )
        self._encoding: ApertusEncoding = encoding
        self._stream: _Projection | None = None
        self._num_calls = 0

    def parse(
        self,
        model_output: str,
        request: Any,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        if model_output and not model_output_token_ids:
            raise ValueError("Apertus 2 parsing requires the generated token IDs")
        projection = _Projection(self._encoding, _history_call_ids(request))
        delta = projection.push(list(model_output_token_ids))
        delta.extend(projection.finish())
        _without_disabled_calls(delta, request)
        calls = [
            FunctionCall(id=call_id, name=name, arguments=arguments)
            for call_id, name, arguments in delta.calls
        ]
        return delta.reasoning or None, delta.content or None, calls or None

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: Any,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        if self._stream is None:
            self._stream = _Projection(self._encoding, _history_call_ids(request))
        delta = self._stream.push(delta_token_ids)
        if finished:
            delta.extend(self._stream.finish())
        _without_disabled_calls(delta, request)
        fields: dict[str, Any] = {}
        if delta.reasoning and getattr(request, "include_reasoning", True):
            fields["reasoning"] = delta.reasoning
        if delta.content:
            fields["content"] = delta.content
        if delta.calls:
            fields["tool_calls"] = [
                DeltaToolCall(
                    index=self._num_calls + i,
                    id=call_id,
                    type="function",
                    function=DeltaFunctionCall(name=name, arguments=arguments),
                )
                for i, (call_id, name, arguments) in enumerate(delta.calls)
            ]
            self._num_calls += len(delta.calls)
        return DeltaMessage(**fields) if fields else None

    def adjust_request(self, request: Any) -> Any:
        request = super().adjust_request(request)
        self._reject_unsupported_options(request)
        forbidden = self._encoding.generation_constraints.forbidden_token_ids
        # Merged last, so client biases cannot re-enable these tokens.
        request.logit_bias = {
            **(request.logit_bias or {}),
            **{str(token_id): -100.0 for token_id in forbidden},
        }
        return request

    def _reject_unsupported_options(self, request: Any) -> None:
        """Refuse options that bypass the wait stop or the forbidden-token mask."""
        special = self._encoding.special_token_ids
        reasons = []
        if getattr(request, "ignore_eos", False):
            reasons.append("ignore_eos would run past <|wait|>")
        if getattr(request, "min_tokens", 0):
            reasons.append("min_tokens would suppress <|wait|>")
        if getattr(request, "use_beam_search", False):
            reasons.append("beam search skips the forbidden-token mask")
        if getattr(request, "stop", None):
            reasons.append(
                "stop strings would also match headers and thinking; omit stop"
            )
        if getattr(request, "tool_choice", None) not in (None, "auto", "none"):
            reasons.append(
                "forced tool calls are unsupported; use tool_choice auto or none"
            )
        response_format = getattr(request, "response_format", None)
        text_format = getattr(getattr(request, "text", None), "format", None)
        if getattr(request, "structured_outputs", None) is not None or any(
            getattr(fmt, "type", "text") != "text"
            for fmt in (response_format, text_format)
        ):
            reasons.append("constrained outputs would force unframed text")
        if getattr(request, "truncate_prompt_tokens", None) is not None or (
            hasattr(request, "truncation") and request.truncation != "disabled"
        ):
            reasons.append("token-level truncation can cut inside a message")
        if _special_ids_in_bias(getattr(request, "logit_bias", None), special):
            reasons.append("logit_bias must not touch special tokens")
        if set(getattr(request, "stop_token_ids", None) or ()) - {
            self._encoding.controls.wait
        }:
            reasons.append("stop_token_ids would end a burst before <|wait|>")
        if getattr(request, "allowed_token_ids", None):
            reasons.append("allowed_token_ids could exclude the control tokens")
        if any(
            special.intersection(
                self.model_tokenizer.encode(word, add_special_tokens=False)
            )
            for word in getattr(request, "bad_words", None) or ()
        ):
            reasons.append("bad_words must not contain special tokens")
        if reasons:
            raise ValueError(f"Unsupported for Apertus 2: {'; '.join(reasons)}")

    # ── Reasoning state from token IDs (structured outputs, usage) ─────

    def _kind(self, token_ids: Sequence[int]) -> str:
        try:
            return self._encoding.decode_text(list(token_ids)).strip()
        except TokenizerError:
            return ""

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        controls = self._encoding.controls
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] != controls.output_start:
                continue
            kind_ids: list[int] = []
            for token_id in input_ids[i + 1 :]:
                if token_id == controls.header_end:
                    break
                kind_ids.append(token_id)
            if self._kind(kind_ids) == "think":
                return controls.output_end in input_ids[i + 1 :]
            return True
        return False

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        controls = self._encoding.controls
        count = 0
        i = 0
        while i < len(token_ids):
            if token_ids[i] != controls.output_start:
                i += 1
                continue
            j = i + 1
            while j < len(token_ids) and token_ids[j] != controls.header_end:
                j += 1
            k = j + 1
            while k < len(token_ids) and token_ids[k] != controls.output_end:
                k += 1
            if self._kind(token_ids[i + 1 : j]) == "think":
                count += max(0, k - j - 1)
            i = k + 1
        return count


class Apertus2ReasoningParser(ParserEngineReasoningAdapter):
    """Registered so vLLM finds :class:`Apertus2Parser`; never parses alone."""

    _parser_engine_cls = Apertus2Parser

    def adjust_request(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError(_FLAGS)

    def extract_reasoning(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError(_FLAGS)

    def extract_reasoning_streaming(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError(_FLAGS)


class Apertus2ToolParser(ParserEngineToolAdapter):
    """Registered so vLLM finds :class:`Apertus2Parser`; never parses alone."""

    _parser_engine_cls = Apertus2Parser

    def adjust_request(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError(_FLAGS)

    def extract_tool_calls(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError(_FLAGS)

    def extract_tool_calls_streaming(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError(_FLAGS)


# The serving layer finds the adapters here; with both flags it calls
# Apertus2Parser.adjust_request, otherwise the adapters refuse the request.
Apertus2Parser.reasoning_parser_cls = Apertus2ReasoningParser  # type: ignore[attr-defined]
Apertus2Parser.tool_parser_cls = Apertus2ToolParser  # type: ignore[attr-defined]

__all__ = [
    "Apertus2Parser",
    "Apertus2ReasoningParser",
    "Apertus2ToolParser",
]
