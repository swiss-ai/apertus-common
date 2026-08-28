"""Wire IR and serialization projections of the typed model.

``render_str`` renders a ``Conversation`` to the format's string form.
The wire IR and the lexer and output machine consumed by the public
``StreamParser`` facade stay internal. The token projection is planned.
"""

from apertus_common.core._render import render_str

__all__ = [
    "render_str",
]
