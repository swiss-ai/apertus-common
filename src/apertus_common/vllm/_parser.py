"""vLLM reasoning and tool parser for Apertus 2 output frames.

Generated text (with special tokens kept) looks like::

    <|out|>think<|hdr|>...reasoning...<|/out|>
    <|out|>reply<|hdr|>...visible text...<|/out|>
    <|out|>call{"name":"get_weather","counter":-1}<|hdr|>{"city":"SF"}<|/out|>
    <|wait|>

The frames drive one declarative :class:`ParserEngineConfig` state machine
(as ``vllm.parser.inkling`` does for a similar block format): think frames
become ``reasoning``, reply frames ``content`` and call frames ``tool_calls``.
Frame kinds the chat API has no channel for (``claim``, unknown kinds) are
passed through as content. ``<|/out|>`` closes every frame, so it carries the
``THINK_END`` label the engine keys its reasoning plumbing on.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

from apertus_common.core._wire import ControlToken

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

OUT = ControlToken.OUT.value
OUT_END = ControlToken.OUT_END.value
HDR = ControlToken.HDR.value
WAIT = ControlToken.WAIT.value

THINK_START = f"{OUT}think{HDR}"
REPLY_START = f"{OUT}reply{HDR}"
CALL_START = f"{OUT}call"


def _passthrough_args(raw_args: str, partial: bool) -> str:
    """The call payload already is the JSON arguments.

    A converter is still needed: without one the engine streams raw chunks and
    loses the chunk that arrives together with the tool name, whereas with one
    it diffs the accumulated arguments against what it has streamed.
    """
    return raw_args


@functools.cache
def apertus2_config() -> ParserEngineConfig:
    terminals = {
        "THINK_START": THINK_START,
        "REPLY_START": REPLY_START,
        "CALL_START": CALL_START,
        "OUT": OUT,
        "HDR": HDR,
        "THINK_END": OUT_END,
        "WAIT": WAIT,
    }
    reasoning_end = (EventType.REASONING_END,)
    tool_start = (EventType.REASONING_END, EventType.TOOL_CALL_START)
    transitions: dict[tuple[ParserState, str], Transition] = {
        # Frame openers. A reply or call opener proves no reasoning is open,
        # which is what lets the serving layer hand later frames to the tool
        # pass (see the same comment in vllm.parser.inkling).
        (ParserState.CONTENT, "THINK_START"): Transition(
            ParserState.REASONING, (EventType.REASONING_START,)
        ),
        (ParserState.CONTENT, "REPLY_START"): Transition(
            ParserState.CONTENT, reasoning_end
        ),
        (ParserState.CONTENT, "CALL_START"): Transition(
            ParserState.TOOL_NAME, tool_start
        ),
        # Any other kind: header text is dropped, the payload is content.
        (ParserState.CONTENT, "OUT"): Transition(ParserState.MESSAGE_HEADER, ()),
        (ParserState.MESSAGE_HEADER, "HDR"): Transition(
            ParserState.CONTENT, reasoning_end
        ),
        # Defensive: a frame opening while a think frame is still unclosed.
        (ParserState.REASONING, "THINK_START"): Transition(ParserState.REASONING, ()),
        (ParserState.REASONING, "REPLY_START"): Transition(
            ParserState.CONTENT, reasoning_end
        ),
        (ParserState.REASONING, "CALL_START"): Transition(
            ParserState.TOOL_NAME, tool_start
        ),
        (ParserState.REASONING, "OUT"): Transition(
            ParserState.MESSAGE_HEADER, reasoning_end
        ),
        # The call header {"name":...,"counter":...} ends at <|hdr|>; the
        # payload (JSON arguments) follows.
        (ParserState.TOOL_NAME, "HDR"): Transition(ParserState.TOOL_ARGS, ()),
        # Stray <|hdr|> in text is dropped.
        (ParserState.CONTENT, "HDR"): Transition(ParserState.CONTENT, ()),
    }
    # <|/out|> and the <|wait|> end of turn close whatever frame is open.
    for end in ("THINK_END", "WAIT"):
        transitions[(ParserState.CONTENT, end)] = Transition(ParserState.CONTENT, ())
        transitions[(ParserState.REASONING, end)] = Transition(
            ParserState.CONTENT, reasoning_end
        )
        transitions[(ParserState.MESSAGE_HEADER, end)] = Transition(
            ParserState.CONTENT, ()
        )
        for tool_state in (ParserState.TOOL_NAME, ParserState.TOOL_ARGS):
            transitions[(tool_state, end)] = Transition(
                ParserState.CONTENT, (EventType.TOOL_CALL_END,)
            )
    return ParserEngineConfig(
        name="apertus2",
        initial_state=ParserState.CONTENT,
        terminals=terminals,
        # Text grammar only: the frame kind is ordinary text between two
        # control tokens, so token-ID matching cannot tell the frames apart.
        token_id_terminals={},
        transitions=transitions,
        arg_converter=_passthrough_args,
        stream_arg_deltas=True,
        tool_args_json=True,
        strip_trailing_reasoning_whitespace=True,
        drop_whitespace_only_content_before_tools=True,
        strip_content_whitespace_with_tools=False,
        validate_tool_names=False,
    )


class Apertus2Parser(ParserEngine):
    CONFIG_NAME = "apertus2"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("parser_engine_config", apertus2_config())
        super().__init__(tokenizer, tools, **kwargs)
        # Raw call-frame headers per tool index, until their JSON is complete.
        self._call_headers: dict[int, str] = {}

    def _reset(self, initial_state: ParserState | None = None) -> None:
        super()._reset(initial_state)
        self._call_headers = {}

    def _handle_tool_name(self, event: SemanticEvent) -> None:
        # The TOOL_NAME text is the frame header JSON, streamed in pieces;
        # the function name is its "name" field.
        idx = event.tool_index
        header = self._call_headers.get(idx, "") + event.value
        self._call_headers[idx] = header
        try:
            parsed = json.loads(header)
        except json.JSONDecodeError:
            return
        if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
            self._tool_slots[idx].name = parsed["name"]

    # ── Reasoning state from token IDs (structured outputs) ─────────────

    def _control_id(self, glyph: str) -> int | None:
        return self.vocab.get(glyph)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        out_id = self._control_id(OUT)
        hdr_id = self._control_id(HDR)
        end_id = self._control_id(OUT_END)
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] != out_id:
                continue
            kind_ids: list[int] = []
            for token_id in input_ids[i + 1 :]:
                if token_id == hdr_id:
                    break
                kind_ids.append(token_id)
            kind = self.model_tokenizer.decode(kind_ids).strip()
            if kind == "think":
                return end_id in input_ids[i + 1 :]
            return True
        return False

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        out_id = self._control_id(OUT)
        hdr_id = self._control_id(HDR)
        end_id = self._control_id(OUT_END)
        count = 0
        i = 0
        while i < len(token_ids):
            if token_ids[i] != out_id:
                i += 1
                continue
            j = i + 1
            while j < len(token_ids) and token_ids[j] != hdr_id:
                j += 1
            kind = self.model_tokenizer.decode(token_ids[i + 1 : j]).strip()
            k = j + 1
            while k < len(token_ids) and token_ids[k] != end_id:
                k += 1
            if kind == "think":
                count += max(0, k - j - 1)
            i = k + 1
        return count


Apertus2ReasoningParser, Apertus2ToolParser = make_adapters(Apertus2Parser)

__all__ = [
    "Apertus2Parser",
    "Apertus2ReasoningParser",
    "Apertus2ToolParser",
    "apertus2_config",
]
