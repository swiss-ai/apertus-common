# Module structure and naming conventions

Three nested scopes, one rule each. The pattern follows PEP 8, the
[typing spec's re-export rules](https://typing.python.org/en/latest/spec/distributing.html#import-conventions),
and the layout of httpx / pydantic v2. Single underscore only, at every
level; no `__` name-mangling.

## 1. Library level: the public API

The public API is exactly what the `__init__.py` files list in
`__all__`; nothing else is a contract.

- **Re-export is recursive:** every public name of a subpackage is also
  re-exported by its parent, so the flat top level
  (`from apertus_common import User`) always offers the whole API.
  Concretely: `__all__` of `apertus_common.model` is a subset of
  `__all__` of `apertus_common`.
- Re-exports are explicit named imports, never `import *`; presence in
  `__all__` (sorted) is what marks a name as intentionally public, for
  readers and for pyright/mypy alike. `tests/test_public_api.py` guards
  the surface.
- `__init__.py` files contain only docstring, re-exports, and `__all__`.
- One `py.typed` at the package root (`src/apertus_common/`); PEP 561
  marks the whole distribution as typed, subpackages never carry a copy.

## 2. Package level: filenames

Implementation modules are underscore-prefixed (`model/_base.py`).
Users import from the package, never from a `_*` path, so files can be
split, merged, or renamed without a breaking change.

## 3. Module level: names inside a `_*` file

The module underscore already makes everything in the file non-public;
the inner underscore is a second, finer signal:

- **Unprefixed** (`User`, `freeze_list`, `is_registered`): the module's
  export surface, meaning re-exported publicly via `__init__.py` or
  shared with sibling modules. Staying out of `__all__` is what keeps
  shared-only helpers package-private.
- **Underscored** (`_coerce_payload`, `_REGISTRY`, `_MessageBase`):
  local to this module; siblings must not import it.

Rule of thumb: a sibling needs it, drop the underscore and keep it out
of `__all__`; only this module needs it, keep the underscore.

## Adding a public symbol

Define it unprefixed in its `_*` module, add it to the `__init__.py`
and `__all__` of its package, mirror it in every parent `__all__` up to
the top level, and update `tests/test_public_api.py`.