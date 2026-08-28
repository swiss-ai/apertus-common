"""Streaming output parsing: the serving reverse path.

Re-exports the public parsing surface implemented in the private
``_stream`` module. The parser's exception hierarchy and the stable
:class:`~apertus_common.errors.ParseFailure` reasons live in
:mod:`apertus_common.errors`.
"""

from apertus_common.parsing._stream import PayloadDelta, StreamParser

__all__ = [
    "PayloadDelta",
    "StreamParser",
]
