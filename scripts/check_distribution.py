"""Install a built wheel in isolation and smoke-test its base and HF extras."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tarfile
from email import message_from_bytes
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def check_artifacts(wheel: Path, source: Path) -> None:
    with ZipFile(wheel) as archive:
        path = next(name for name in archive.namelist() if name.endswith("/METADATA"))
        metadata = message_from_bytes(archive.read(path))
        assert metadata.get("Description-Content-Type") == "text/markdown"
        description = metadata.get_payload()
        assert isinstance(description, str) and description.strip()
        assert metadata.get_all("Project-URL")
    with tarfile.open(source) as archive:
        for member in archive.getnames():
            parts = Path(member).parts
            assert ".claude" not in parts
            assert parts[-1] not in {"AGENTS.md", "CLAUDE.md", "DEVELOPMENT.md"}


def smoke(hf: bool) -> None:
    from apertus_common import (
        Conversation,
        Reply,
        SystemPrompt,
        User,
        Wait,
        from_openai,
        parse_output_text,
        to_openai,
    )

    assert files("apertus_common").joinpath("py.typed").is_file()
    conversation = Conversation(
        system=SystemPrompt.build(),
        items=[User(payload="Hello"), Reply(payload="Hi"), Wait()],
    )
    assert Conversation.from_json(conversation.to_json()) == conversation
    report = conversation.check()
    assert report.ok and report.complete
    assert parse_output_text("<|out|>reply<|hdr|>Hi<|/out|><|wait|>").items == [
        Reply(payload="Hi"),
        Wait(),
    ]
    mapped = from_openai([{"role": "user", "content": "Hello"}])
    assert to_openai(mapped.value, context=mapped.context).value.get("messages") == [
        {"role": "user", "content": "Hello"}
    ]
    if hf:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        assert callable(AutoTokenizer)
        assert callable(PreTrainedTokenizerFast)
    else:
        assert importlib.util.find_spec("transformers") is None
    print(f"Installed wheel smoke passed ({'hf' if hf else 'base'})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", choices=("base", "hf"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.smoke:
        smoke(args.smoke == "hf")
        return
    wheel = Path(
        f"dist/apertus_common-{version('apertus-common')}-py3-none-any.whl"
    ).resolve()
    if not wheel.is_file():
        parser.error(f"missing wheel: {wheel}; run make build first")
    check_artifacts(
        wheel, wheel.with_name(f"apertus_common-{version('apertus-common')}.tar.gz")
    )
    script = Path(__file__).resolve()
    with TemporaryDirectory(prefix="apertus-distribution-") as directory:
        environment = Path(directory) / ".venv"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(environment)],
            cwd=directory,
            check=True,
        )
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        for extra, mode in (("", "base"), ("[hf]", "hf")):
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python), f"{wheel}{extra}"],
                cwd=directory,
                check=True,
            )
            subprocess.run(
                [str(python), "-I", str(script), "--smoke", mode],
                cwd=directory,
                check=True,
            )


if __name__ == "__main__":
    main()
