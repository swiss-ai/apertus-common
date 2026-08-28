"""Ingest and export: the model's own JSON and OpenAI messages.

``from_json``/``to_json`` are harmony-style names for the canonical model
JSON (``Conversation.model_validate_json``/``model_dump_json``);
``from_openai``/``to_openai`` map OpenAI chat-completions messages, a lossy
adapter whose subset is stated on ``to_openai``. Both lifts take the ingest
flags: ``normalize=True`` applies canonical input order, ``check=True``
returns ``(conversation, report)``, ``strict=True`` raises
``ConformanceError`` on error-severity findings (warnings pass). A
JSON-lines dataset is one ``to_json`` per line, ``from_json`` per line.
"""

from apertus_common.formats._json import from_json, to_json
from apertus_common.formats._openai import from_openai, to_openai

__all__ = [
    "from_json",
    "from_openai",
    "to_json",
    "to_openai",
]
