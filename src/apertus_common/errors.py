"""Shared exception hierarchy for invalid Apertus format data.

The model package raises :class:`HeaderError` from Pydantic validators when a
header cannot be represented safely. Callers normally receive it inside a
``pydantic.ValidationError``, while direct validation helpers may expose it as
an :class:`ApertusFormatError`. Parser- and vocabulary-specific subclasses will
join this module when the codec and conformance layers are implemented.
"""


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
