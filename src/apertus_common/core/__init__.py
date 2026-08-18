"""Wire IR and serialization projections of the typed model.

``render_str`` renders a ``Conversation`` to the format's string form.
The wire IR and the token projection stay internal until the parser and
the tokenizer binding land.
"""

from apertus_common.core._render import render_str

__all__ = [
    "render_str",
]
