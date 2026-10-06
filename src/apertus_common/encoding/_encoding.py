"""Verified tokenizer binding and a token frontend for the shared parser."""

from __future__ import annotations

import codecs
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from apertus_common.core._lexeme import ControlLexeme, Lexeme, SourceSpan, TextLexeme
from apertus_common.core._render import to_wire
from apertus_common.core._wire import Control, ControlToken
from apertus_common.errors import ParseFailure, ParserStateError, TokenizerError
from apertus_common.model import Conversation
from apertus_common.model._conversation import ConversationItem
from apertus_common.parsing import ParseBoundary
from apertus_common.parsing._stream import ParserCore

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

_CONTROL_ROLES = {
    ControlToken.IN: "input_start_token",
    ControlToken.IN_END: "input_end_token",
    ControlToken.OUT: "output_start_token",
    ControlToken.OUT_END: "output_end_token",
    ControlToken.HDR: "header_end_token",
    ControlToken.WAIT: "wait_token",
    ControlToken.PAD: "pad_token",
}


@dataclass(frozen=True, slots=True)
class ControlTokenIds:
    """Artifact IDs for the seven format controls."""

    input_start: int
    input_end: int
    output_start: int
    output_end: int
    header_end: int
    wait: int
    pad: int


@dataclass(frozen=True, slots=True)
class GenerationConstraints:
    """Engine configuration; per-message request suspension is separate."""

    eos_token_id: int
    forbidden_token_ids: tuple[int, ...]


def _byte_alphabet() -> dict[str, int]:
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    result = {chr(value): value for value in values}
    extra = 0
    for value in range(256):
        if value not in values:
            result[chr(256 + extra)] = value
            extra += 1
    return result


_BYTE_ALPHABET = _byte_alphabet()


class ApertusEncoding:
    """A text-preserving snapshot of an HF fast tokenizer and its named controls."""

    def __init__(self, artifact: PreTrainedTokenizerFast | str | Path) -> None:
        try:
            self._load(artifact)
        except TokenizerError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            raise TokenizerError(f"invalid tokenizer artifact: {error}") from error

    def _load(self, artifact: PreTrainedTokenizerFast | str | Path) -> None:
        from tokenizers import Tokenizer

        if isinstance(artifact, (str, Path)):
            from transformers import AutoTokenizer

            tokenizer = cast(
                "PreTrainedTokenizerFast",
                cast(Any, AutoTokenizer).from_pretrained(
                    artifact, local_files_only=True
                ),
            )
        else:
            tokenizer = artifact
        if not tokenizer.is_fast or not isinstance(
            cast(object, tokenizer.backend_tokenizer), Tokenizer
        ):
            raise TypeError("an HF fast tokenizer is required")
        state: dict[str, Any] = json.loads(tokenizer.backend_tokenizer.to_str())
        if state.get("normalizer") is not None:
            raise ValueError("the tokenizer must not normalize text")
        if state.get("decoder", {}).get("type") != "ByteLevel":
            raise ValueError("this profile binding requires the ByteLevel decoder")
        # Snapshot the backend: caller padding, truncation and postprocessing
        # must neither alter content nor be changed by this binding.
        state.update(post_processor=None, padding=None, truncation=None)
        self._validate_backend(state)
        try:
            backend = Tokenizer.from_str(json.dumps(state, ensure_ascii=False))
        except Exception as error:  # Rust binding exposes a generic exception.
            raise TokenizerError(f"invalid tokenizer backend: {error}") from error
        special_ids = frozenset(
            i
            for i, token in backend.get_added_tokens_decoder().items()
            if token.special
        )
        controls: dict[str, int] = {}
        for control, role in _CONTROL_ROLES.items():
            glyph = getattr(tokenizer, role, None)
            token_id = getattr(tokenizer, role + "_id", None)
            if (
                not isinstance(glyph, str)
                or glyph != control.value
                or type(token_id) is not int
                or backend.token_to_id(glyph) != token_id
                or token_id not in special_ids
            ):
                raise ValueError(
                    f"{role} must name the registered special {control.value}"
                )
            controls[glyph] = token_id
        if len(set(controls.values())) != len(_CONTROL_ROLES):
            raise ValueError("control IDs must be distinct integers")
        # True disables the added-special matcher; add_special_tokens=False
        # alone only disables postprocessing and does not protect payloads.
        backend.encode_special_tokens = True
        self._backend: Tokenizer = backend
        self._vocab_size = backend.get_vocab_size(with_added_tokens=True)
        self._special_ids = special_ids
        self._control_ids = {
            ControlToken(glyph): token_id for glyph, token_id in controls.items()
        }
        self._bytes: dict[int, bytes] = {}
        self._controls = ControlTokenIds(*(self._control_ids[t] for t in ControlToken))
        allowed = {
            self._controls.output_start,
            self._controls.output_end,
            self._controls.header_end,
            self._controls.wait,
        }
        self._generation_constraints = GenerationConstraints(
            self._controls.wait, tuple(sorted(special_ids - allowed))
        )
        for probe in [
            *(t.value for t in ControlToken),
            "e\u0301",
            "\r\n",
            " <image> <audio> ",
            "\x00",
            "\ufffd🙂",
        ]:
            ids = self.encode_text(probe)
            if self.decode_text(ids) != probe:
                raise ValueError("artifact ordinary-text path is not lossless")

    @staticmethod
    def _validate_backend(state: dict[str, Any]) -> None:
        def object_value(value: object) -> dict[str, Any]:
            if not isinstance(value, dict):
                raise TypeError("expected a tokenizer configuration object")
            return cast(dict[str, Any], value)

        def array_value(value: object) -> list[object]:
            if not isinstance(value, list):
                raise TypeError("expected a tokenizer configuration array")
            return cast("list[object]", value)

        pre = object_value(state.get("pre_tokenizer"))
        parts = (
            array_value(pre.get("pretokenizers"))
            if pre.get("type") == "Sequence"
            else [pre]
        )
        if not parts:
            raise ValueError("unsupported pretokenizer")
        for value in parts[:-1]:
            part = object_value(value)
            if (
                part.get("type") != "Split"
                or part.get("behavior") != "Isolated"
                or part.get("invert") is not False
            ):
                raise ValueError("pretokenizer splits must retain every character")
        byte_level = object_value(parts[-1])
        if (
            byte_level.get("type") != "ByteLevel"
            or byte_level.get("add_prefix_space") is not False
        ):
            raise ValueError(
                "pretokenizer must end in ByteLevel without prefix insertion"
            )
        model = object_value(state.get("model"))
        if (
            model.get("type") != "BPE"
            or any(
                model.get(key) is not None
                for key in (
                    "dropout",
                    "unk_token",
                    "continuing_subword_prefix",
                    "end_of_word_suffix",
                )
            )
            or any(
                model.get(key, False)
                for key in ("fuse_unk", "byte_fallback", "ignore_merges")
            )
        ):
            raise ValueError("unsupported or text-rewriting BPE configuration")
        vocab = object_value(model.get("vocab"))
        if any(type(i) is not int for i in vocab.values()) or set(
            vocab.values()
        ) != set(range(len(vocab))):
            raise ValueError(
                "vocabulary IDs must be unique contiguous nonnegative integers"
            )
        added = [
            object_value(value) for value in array_value(state.get("added_tokens"))
        ]
        inventory = dict(vocab)
        seen: set[int] = set()
        for token in added:
            token_id, content = token.get("id"), token.get("content")
            if (
                type(token_id) is not int
                or not isinstance(content, str)
                or token_id in seen
            ):
                raise ValueError("added token IDs/content must be unique and valid")
            if content in inventory and inventory[content] != token_id:
                raise ValueError("added token IDs/content must match the vocabulary")
            seen.add(token_id)
            inventory[content] = token_id
            if token.get("special") is not True:
                raise ValueError("ordinary added-token matching is not supported")
        if set(inventory.values()) != set(range(len(inventory))):
            raise ValueError("combined vocabulary IDs must be unique and contiguous")
        if any(char not in vocab or vocab[char] in seen for char in _BYTE_ALPHABET):
            raise ValueError("the ordinary byte alphabet must be complete")
        for merge in array_value(model.get("merges")):
            pair = merge.split(" ") if isinstance(merge, str) else array_value(merge)
            if len(pair) != 2 or not all(isinstance(part, str) for part in pair):
                raise ValueError("BPE merges must contain two vocabulary strings")
            if vocab.get("".join(cast(list[str], pair))) in seen:
                raise ValueError("ordinary BPE merges must not produce special IDs")
        if any(
            any(char not in _BYTE_ALPHABET for char in token)
            for token, token_id in vocab.items()
            if token_id not in seen
        ):
            raise ValueError("ordinary vocabulary must use the ByteLevel alphabet")

    @property
    def controls(self) -> ControlTokenIds:
        return self._controls

    @property
    def generation_constraints(self) -> GenerationConstraints:
        return self._generation_constraints

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def special_token_ids(self) -> frozenset[int]:
        return self._special_ids

    def get_vocab(self) -> dict[str, int]:
        return self._backend.get_vocab(with_added_tokens=True)

    def token_to_id(self, token: str) -> int | None:
        return self._backend.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        if type(token_id) is not int or token_id < 0 or token_id >= self.vocab_size:
            return None
        return self._backend.id_to_token(token_id)

    def encode_text(self, text: str) -> list[int]:
        if not isinstance(cast(object, text), str) or any(
            0xD800 <= ord(char) <= 0xDFFF for char in text
        ):
            raise TokenizerError("text must contain Unicode scalar values")
        ids = self._backend.encode(text, add_special_tokens=False).ids
        if not self._special_ids.isdisjoint(ids):
            raise TokenizerError("ordinary text encoded to a special ID")
        return ids

    def decode_text(self, token_ids: Sequence[int]) -> str:
        ids = self._validate_ids(token_ids)
        return self._backend.decode(ids, skip_special_tokens=False)

    def _validate_ids(self, ids: Sequence[int]) -> list[int]:
        values = list(ids)
        if any(type(i) is not int or i < 0 or i >= self.vocab_size for i in values):
            raise TokenizerError("unknown or invalid token ID")
        return values

    def encode_conversation(self, conversation: Conversation) -> list[int]:
        ids: list[int] = []
        text: list[str] = []
        for atom in to_wire(conversation):
            if isinstance(atom, Control):
                if text:
                    ids.extend(self.encode_text("".join(text)))
                    text.clear()
                ids.append(self._control_ids[atom.token])
            else:
                text.append(atom.text)
        if text:
            ids.extend(self.encode_text("".join(text)))
        return ids

    def encode_items(self, items: Iterable[ConversationItem]) -> list[int]:
        return self.encode_conversation(Conversation(items=list(items)))

    def output_parser(self) -> TokenStreamParser:
        return TokenStreamParser(self)

    def parse_output(self, token_ids: Sequence[int]) -> Conversation:
        parser = self.output_parser()
        parser.push_tokens(token_ids)
        return parser.finish()

    def token_bytes(self, token_id: int) -> bytes:
        """Decode an ordinary ByteLevel vocabulary entry without UTF-8 repair."""
        if token_id < 0 or token_id >= self.vocab_size:
            raise TokenizerError("unknown token ID")
        if token_id not in self._bytes:
            token = self._backend.id_to_token(token_id)
            if token is None:
                raise TokenizerError("unknown token ID")
            try:
                self._bytes[token_id] = bytes(_BYTE_ALPHABET[char] for char in token)
            except KeyError as error:
                raise TokenizerError(
                    "ordinary token is not ByteLevel encoded"
                ) from error
        return self._bytes[token_id]


class TokenStreamParser(ParserCore):
    """Incremental ID parsing; decoded payload glyphs never become controls."""

    def __init__(self, encoding: ApertusEncoding) -> None:
        super().__init__("token")
        self._encoding = encoding
        self._control_ids = {
            encoding.token_to_id(control.value): control for control in ControlToken
        }
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._position = 0
        self._pending_start = 0

    def push_tokens(self, token_ids: Sequence[int]) -> None:
        self.ensure_active()
        self.feed(self._lexemes(token_ids))

    def _lexemes(self, token_ids: Sequence[int]) -> Iterator[Lexeme]:
        pieces: list[str] = []
        start = self._pending_start
        for token_id in token_ids:
            position = self._position
            if self.boundary is ParseBoundary.WAIT:
                self.fail(
                    ParseFailure.TRAILING_AFTER_WAIT, position, "input after wait"
                )
            if type(token_id) is not int or token_id < 0:
                self.fail(ParseFailure.INVALID_ENCODING, position, "invalid token ID")
            control = self._control_ids.get(token_id)
            if token_id in self._encoding.special_token_ids:
                if pieces:
                    yield TextLexeme(
                        "".join(pieces), SourceSpan(start, position, "token")
                    )
                    pieces.clear()
                if self._decoder.getstate()[0]:
                    self.fail(
                        ParseFailure.INVALID_ENCODING,
                        position,
                        "incomplete UTF-8 before control",
                    )
                if control is None:
                    self.fail(
                        ParseFailure.UNEXPECTED_CONTROL,
                        position,
                        "forbidden special token",
                    )
                self._position += 1
                yield ControlLexeme(
                    control, SourceSpan(position, position + 1, "token")
                )
                start = self._pending_start = self._position
            else:
                try:
                    value = self._decoder.decode(self._encoding.token_bytes(token_id))
                except (TokenizerError, UnicodeDecodeError) as error:
                    if pieces:
                        yield TextLexeme(
                            "".join(pieces), SourceSpan(start, position, "token")
                        )
                    self.fail(
                        ParseFailure.INVALID_ENCODING, position, str(error), cause=error
                    )
                self._position += 1
                pieces.append(value)
        if pieces:
            yield TextLexeme(
                "".join(pieces), SourceSpan(start, self._position, "token")
            )
        self._pending_start = self._position

    def finish(self) -> Conversation:
        if self._failed:
            raise ParserStateError("finish on a failed parser; use a new parser")
        if self._result is not None:
            return self._result
        if self._decoder.getstate()[0]:
            self.fail(
                ParseFailure.INVALID_ENCODING, self._position, "incomplete UTF-8 at end"
            )
        return super().finish()


def load_encoding(artifact: PreTrainedTokenizerFast | str | Path) -> ApertusEncoding:
    """Bind a loaded HF fast tokenizer, or load a local directory with the hf extra."""
    return ApertusEncoding(artifact)
