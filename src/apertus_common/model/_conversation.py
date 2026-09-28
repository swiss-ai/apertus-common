"""Root conversation model, item dispatch, and canonical normalization.

This module assembles the system, input, and output models into the public
:class:`Conversation` entry point. Its callable discriminator selects items by
``(direction, type)``, handles :class:`Wait` separately, and routes unknown
types to the generic model for their direction. As a result, validating Python
or JSON data restores exact concrete classes. ``normalized()`` uses metadata
from ``_base``; the checker package reports broader conversation rules.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from apertus_common.model._base import (
    FrozenList,
    ImmutableList,
    InputMessage,
    is_registered,
    type_info,
)
from apertus_common.model._input import (
    Attachment,
    Document,
    Event,
    GenericInput,
    Harness,
    Retrieval,
    ToolResult,
    User,
)
from apertus_common.model._output import (
    Assistant,
    GenericOutput,
    Think,
    ToolCall,
    VerifiableAnswer,
)
from apertus_common.model._system import SystemPrompt


class Wait(BaseModel):
    """Typed representation of the model's deliberate ``<|wait|>`` stop.

    ``Wait`` participates directly in :class:`Conversation.items` but is not an
    input or output message. Its ``type`` field exists only for
    JSON discrimination; the renderer emits a bare control token.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        json_schema_extra={"required": ["type"]},
    )

    type: Literal["wait"] = "wait"


def _item_tag(value: Any) -> str | None:
    if isinstance(value, dict):
        item = cast(dict[object, object], value)
        type_value = item.get("type")
        direction = item.get("direction")
    else:
        type_value = getattr(value, "type", None)
        direction = getattr(value, "direction", None)
    if direction is None:
        return "wait" if type_value == "wait" else None
    if direction not in ("in", "out") or not isinstance(type_value, str):
        return None
    if is_registered(direction, type_value):
        return f"{direction}:{type_value}"
    return f"{direction}:*"


# The callable discriminator turns each item into a lookup tag such as
# ``in:user`` or ``out:tool_call``; Pydantic then validates against the union
# member carrying that ``Tag``. Unknown types use ``in:*``/``out:*`` and wait
# uses its directionless ``wait`` tag.
ConversationItem = Annotated[
    Annotated[Harness, Tag("in:harness")]
    | Annotated[ToolResult, Tag("in:tool_result")]
    | Annotated[Retrieval, Tag("in:retrieval")]
    | Annotated[Event, Tag("in:event")]
    | Annotated[Attachment, Tag("in:attachment")]
    | Annotated[User, Tag("in:user")]
    | Annotated[Document, Tag("in:document")]
    | Annotated[GenericInput, Tag("in:*")]
    | Annotated[Assistant, Tag("out:assistant")]
    | Annotated[Think, Tag("out:think")]
    | Annotated[ToolCall, Tag("out:tool_call")]
    | Annotated[VerifiableAnswer, Tag("out:verifiable_answer")]
    | Annotated[GenericOutput, Tag("out:*")]
    | Annotated[Wait, Tag("wait")],
    Discriminator(_item_tag),
]
ConversationItems = ImmutableList[ConversationItem]


def _empty_conversation_items() -> ConversationItems:
    return FrozenList(())


class _InputWithType(Protocol):
    """Typing view of the input type used for delivery ordering."""

    type: str


def delivery_order(message: InputMessage) -> int:
    """Return the canonical delivery position of an input message.

    Unlabeled types (open vocabulary, ``document``) key at position 3.
    The single definition shared by ``Conversation.normalized()`` and the
    input-order conformance rule.
    """
    message_type = cast(_InputWithType, message).type
    order = type_info(message.direction, message_type).order
    return 3 if order is None else order


class Conversation(BaseModel):
    """Validated root object for an ordered Apertus conversation.

    Construct it directly from typed values, or use ``model_validate`` and
    ``model_validate_json`` as boundary entry points that discriminate raw item
    data into exact message classes. ``system=None`` represents pretraining
    data. Construction deliberately permits cross-message
    issues such as noncanonical order or unmatched tool IDs so the conformance
    checker can report them without losing the original conversation.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, validate_default=True
    )

    system: SystemPrompt | None = None
    items: ConversationItems = Field(default_factory=_empty_conversation_items)

    def normalized(self) -> Self:
        """Return a conversation with canonical input order at each boundary.

        Stably sorts each maximal run of consecutive input messages by
        canonical delivery order; types without an order use position 3.
        Outputs and waits never move. The operation is idempotent and never
        implicit: conformance paths preserve and report observed order.
        Immutable items can safely be shared between the returned conversation
        and its source.
        """
        items: list[ConversationItem] = []
        run: list[InputMessage] = []

        def flush() -> None:
            # run holds only input-cast members of the item union
            items.extend(cast(list[ConversationItem], sorted(run, key=delivery_order)))
            run.clear()

        for item in self.items:
            if isinstance(item, InputMessage):
                run.append(item)
            else:
                flush()
                items.append(item)
        flush()
        return type(self)(system=self.system, items=items)
