"""Shared exception hierarchy for invalid Apertus format data.

The model package raises :class:`HeaderError` from Pydantic validators when a
header cannot be represented safely. Callers normally receive it inside a
``pydantic.ValidationError``, while direct validation helpers may expose it as
an :class:`ApertusFormatError`. The formats package raises
:class:`AdapterError` for what the OpenAI adapter cannot map; the checker's
``ConformanceError`` shares the base. Parser- and vocabulary-specific
subclasses will join this module when their codecs land.
"""

from __future__ import annotations

from typing import Literal


class ApertusFormatError(Exception):
    """Base class for errors caused by invalid Apertus format data.

    Catch this when a caller wants one library-level exception category rather
    than handling a specific header, framing, or vocabulary failure.
    """


class HeaderError(ApertusFormatError, ValueError):
    """A message header violates the Apertus grammar.

    The :class:`ValueError` base lets Pydantic attach the failure to the
    relevant model field and include it in ``pydantic.ValidationError``.
    """


class AdapterError(ApertusFormatError):
    """The OpenAI adapter could not map a message, item, or system prompt.

    ``from_openai`` raises it with ``where="message"`` and the index into the
    OpenAI message list, or ``where="system"`` (index ``-1``) for an invalid
    effort or tools entry; ``to_openai`` with ``where="item"`` and the index
    into ``Conversation.items``, or ``where="system"`` for the system prompt.
    A chained pydantic ``ValidationError`` or JSON error is the ``__cause__``
    where one exists.
    """

    def __init__(
        self, where: Literal["message", "item", "system"], index: int, reason: str
    ) -> None:
        super().__init__(where, index, reason)
        self.where = where
        self.index = index
        self.reason = reason

    def __str__(self) -> str:
        if self.where == "system":
            return f"system prompt: {self.reason}"
        noun = "openai message" if self.where == "message" else "conversation item"
        return f"{noun} {self.index}: {self.reason}"
