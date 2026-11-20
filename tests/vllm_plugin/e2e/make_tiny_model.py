"""Write a tiny random-weight Llama checkpoint with the Apertus 2 tokenizer.

The weights are random, so generations are meaningless: the end-to-end test
checks serving plumbing (plugin loading, rendering, guards, parsing paths), not
model quality. Usage: python make_tiny_model.py <tokenizer_dir> <out_dir>
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def main(tokenizer_dir: Path, out_dir: Path) -> None:
    vocab = json.loads((tokenizer_dir / "tokenizer.json").read_text())
    size = len(vocab["model"]["vocab"])
    size += sum(t["id"] >= size for t in vocab["added_tokens"])
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=12,
        pad_token_id=3,
    )
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(out_dir)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        shutil.copy(tokenizer_dir / name, out_dir / name)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
