"""vLLM integration: serve the Apertus 2 format without a chat template.

Installed as a ``vllm.general_plugins`` entry point (see ``pyproject.toml``),
:func:`register` adds to vLLM's registries:

* tokenizer mode and renderer ``apertus2``: chat requests are mapped with
  :func:`apertus_common.from_openai` and encoded to token IDs with
  :func:`apertus_common.load_encoding`;
* reasoning parser and tool parser ``apertus2``: generated token IDs are parsed
  with apertus-common's ``TokenStreamParser``; think, reply and call messages
  become ``reasoning``, ``content`` and ``tool_calls``, streaming included. The
  parser also rejects unsupported request options and masks forbidden tokens.

Serve with::

    vllm serve <checkpoint> --tokenizer-mode apertus2 \\
        --reasoning-parser apertus2 \\
        --enable-auto-tool-choice --tool-call-parser apertus2

The registration is lazy on the vLLM side: this module imports nothing from
vLLM until vLLM itself loads the plugin.
"""

from __future__ import annotations

MODE = "apertus2"
_PACKAGE = "apertus_common.vllm"


def register() -> None:
    """Entry point for ``vllm.general_plugins``; safe to call repeatedly."""
    from vllm.reasoning import ReasoningParserManager
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm.tool_parsers import ToolParserManager

    TokenizerRegistry.register(MODE, f"{_PACKAGE}._tokenizer", "Apertus2Tokenizer")
    RENDERER_REGISTRY.register(MODE, f"{_PACKAGE}._renderer", "Apertus2Renderer")
    ReasoningParserManager.register_lazy_module(
        MODE, f"{_PACKAGE}._parser", "Apertus2ReasoningParser"
    )
    ToolParserManager.register_lazy_module(
        MODE, f"{_PACKAGE}._parser", "Apertus2ToolParser"
    )


__all__ = ["MODE", "register"]
