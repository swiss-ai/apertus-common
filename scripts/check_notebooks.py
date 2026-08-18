"""Type-check and execute Python cells from every example notebook."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted((ROOT / "examples").glob("*.ipynb")))


def code_cells(path: Path) -> str:
    """Return one Python module containing all code cells from ``path``."""
    document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{path} does not contain a notebook object")

    notebook = cast(dict[str, object], document)
    cells_value = notebook.get("cells")
    if not isinstance(cells_value, list):
        raise TypeError(f"{path} has no cell list")

    source: list[str] = []
    for cell_value in cast(list[object], cells_value):
        if not isinstance(cell_value, dict):
            continue
        cell = cast(dict[str, object], cell_value)
        if cell.get("cell_type") != "code":
            continue
        fragments_value = cell.get("source")
        if not isinstance(fragments_value, list):
            raise TypeError(f"a code cell in {path} has no source list")
        for fragment in cast(list[object], fragments_value):
            if not isinstance(fragment, str):
                raise TypeError(f"a code cell in {path} contains non-text source")
            source.append(fragment)
        source.append("\n")
    return "".join(source)


def main() -> int:
    """Export notebook cells, run both type checkers, then execute each file."""
    if not NOTEBOOKS:
        print("no example notebooks found", file=sys.stderr)
        return 1

    with TemporaryDirectory(prefix="apertus-notebooks-") as temporary:
        paths: list[str] = []
        target_dir = Path(temporary)
        for notebook in NOTEBOOKS:
            target = target_dir / f"{notebook.stem}.py"
            target.write_text(code_cells(notebook), encoding="utf-8")
            paths.append(str(target))

        for command in (("mypy", "--strict"), ("pyright",)):
            result = subprocess.run(
                [*command, *paths],
                cwd=ROOT,
                check=False,
            )
            if result.returncode:
                return result.returncode

        for path in paths:
            execution = subprocess.run(
                [sys.executable, path],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if execution.returncode:
                print(f"notebook execution failed: {path}", file=sys.stderr)
                print(execution.stdout, file=sys.stderr)
                print(execution.stderr, file=sys.stderr)
                return execution.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
