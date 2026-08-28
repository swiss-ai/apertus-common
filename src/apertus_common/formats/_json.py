"""Native JSON aliases and the ingest flags shared by every lift.

``from_json``/``to_json`` are harmony-style names over the model's own
``Conversation.model_validate_json``/``model_dump_json``; the typed model JSON
is the one canonical serialization. The ingest flags (``normalize``,
``check``, ``strict``) are implemented once here (``finish``) and reused by
``from_openai``. A dataset file is the JSON-lines recipe over these two
calls: one conversation JSON per line, ``from_json`` per line.
"""

from __future__ import annotations

from typing import Literal, overload

from apertus_common.checker import ConformanceError, ConformanceReport
from apertus_common.checker import check as check_conversation
from apertus_common.model import Conversation


def finish(
    conversation: Conversation, *, normalize: bool, check: bool, strict: bool
) -> Conversation | tuple[Conversation, ConformanceReport]:
    """Apply the ingest flags to a freshly constructed conversation.

    ``normalize`` applies canonical input order; ``check`` returns
    ``(conversation, report)`` instead of the conversation; ``strict`` runs
    the checker and raises ``ConformanceError`` on error-severity violations
    (warnings pass). Package-internal, shared by every lift.
    """
    if normalize:
        conversation = conversation.normalized()
    if not (check or strict):
        return conversation
    report = check_conversation(conversation)
    if strict and not report.ok:
        raise ConformanceError(report)
    return (conversation, report) if check else conversation


@overload
def from_json(
    text: str | bytes,
    *,
    normalize: bool = True,
    check: Literal[False] = False,
    strict: bool = False,
) -> Conversation: ...


@overload
def from_json(
    text: str | bytes,
    *,
    normalize: bool = True,
    check: Literal[True],
    strict: bool = False,
) -> tuple[Conversation, ConformanceReport]: ...


@overload
def from_json(
    text: str | bytes,
    *,
    normalize: bool = True,
    check: bool,
    strict: bool = False,
) -> Conversation | tuple[Conversation, ConformanceReport]: ...


def from_json(
    text: str | bytes,
    *,
    normalize: bool = True,
    check: bool = False,
    strict: bool = False,
) -> Conversation | tuple[Conversation, ConformanceReport]:
    """Validate JSON *text* (str or bytes) into a ``Conversation``.

    Exactly ``Conversation.model_validate_json(text)``: exact classes are
    restored and a malformed document raises pydantic ``ValidationError``.
    Ingest flags ``normalize``, ``check``, ``strict`` as documented on the
    ``formats`` package. One line of a JSON-lines dataset is one such text.
    """
    return finish(
        Conversation.model_validate_json(text),
        normalize=normalize,
        check=check,
        strict=strict,
    )


def to_json(conversation: Conversation) -> str:
    """The canonical JSON of *conversation*: ``model_dump_json()``.

    Lossless and schema-described (``Conversation.model_json_schema()``);
    ``from_json(to_json(c), normalize=False) == c``.
    """
    return conversation.model_dump_json()
