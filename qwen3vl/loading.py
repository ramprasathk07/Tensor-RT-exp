"""Checkpoint loading for the standalone Qwen3-VL implementation.

The model skeleton is built on the meta device so no random weights are ever
materialised, then real tensors are moved in with ``assign=True``. Loading a
2B checkpoint this way costs one copy instead of an init pass plus a copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .config import Qwen3VLConfig
from .modeling import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
)

_ROTARY_TYPES = (Qwen3VLTextRotaryEmbedding, Qwen3VLVisionRotaryEmbedding)


def find_weight_files(model_dir: Path) -> list[Path]:
    """Return the safetensors shards for a checkpoint dir, in index order."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as fh:
            index = json.load(fh)
        names = sorted(set(index["weight_map"].values()))
        return [model_dir / name for name in names]

    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]

    shards = sorted(model_dir.glob("model-*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"no safetensors weights found in {model_dir}")


def load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for shard in find_weight_files(model_dir):
        state_dict.update(load_file(str(shard), device="cpu"))
    return state_dict


def load_qwen3vl(
    model_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> Qwen3VLForConditionalGeneration:
    """Build the model and load its weights.

    Args:
        model_dir: checkpoint directory holding ``config.json`` and safetensors.
        dtype: dtype to cast weights to. The checkpoint ships bfloat16.
        device: destination device for the loaded weights.
        strict: raise if checkpoint and model key sets disagree.
    """
    model_dir = Path(model_dir)
    device = torch.device(device)
    config = Qwen3VLConfig.from_pretrained(model_dir)

    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)

    state_dict = load_state_dict(model_dir)
    state_dict = {k: v.to(dtype=dtype, device=device) for k, v in state_dict.items()}

    # The released checkpoint omits lm_head.weight because it is tied.
    if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.language_model.embed_tokens.weight"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch. missing={missing[:8]} unexpected={unexpected[:8]}")

    # Non-persistent buffers are absent from the checkpoint, so they are still
    # on the meta device after an assign-load; rebuild them on the real device.
    for module in model.modules():
        if isinstance(module, _ROTARY_TYPES):
            module.reset_buffers(device=device)

    if config.tie_word_embeddings:
        model.tie_weights()

    model.eval()
    model.requires_grad_(False)
    return model
