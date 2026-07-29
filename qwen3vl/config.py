"""Configuration dataclasses for Qwen3-VL.

Mirrors the fields of the HuggingFace `config.json` shipped with
`Qwen/Qwen3-VL-2B-Instruct` so a checkpoint directory can be loaded verbatim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def _filter_known(cls, raw: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the dataclass does not declare (config.json carries extras)."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in known}


@dataclass
class Qwen3VLVisionConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    depth: int = 24
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    hidden_act: str = "gelu_pytorch_tanh"
    deepstack_visual_indexes: tuple[int, ...] = (5, 11, 17)
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_grid_per_side(self) -> int:
        """Side length of the square learned position-embedding table."""
        return int(self.num_position_embeddings**0.5)

    @property
    def spatial_merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Qwen3VLVisionConfig:
        raw = dict(raw)
        if "deepstack_visual_indexes" in raw:
            raw["deepstack_visual_indexes"] = tuple(raw["deepstack_visual_indexes"])
        return cls(**_filter_known(cls, raw))


@dataclass
class Qwen3VLTextConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262_144
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int | None = None
    initializer_range: float = 0.02
    # rope_scaling in config.json -> mrope_section / mrope_interleaved
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    mrope_interleaved: bool = True

    def __post_init__(self) -> None:
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError(
                f"mrope_section {self.mrope_section} must sum to head_dim//2 = {self.head_dim // 2}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Qwen3VLTextConfig:
        raw = dict(raw)
        rope_scaling = raw.pop("rope_scaling", None) or {}
        if "mrope_section" in rope_scaling:
            raw["mrope_section"] = tuple(rope_scaling["mrope_section"])
        if "mrope_interleaved" in rope_scaling:
            raw["mrope_interleaved"] = bool(rope_scaling["mrope_interleaved"])
        rope_type = rope_scaling.get("rope_type", "default")
        if rope_type != "default":
            raise NotImplementedError(f"only 'default' rope_type is supported, got {rope_type!r}")
        # eos_token_id may be a list in generation configs
        if isinstance(raw.get("eos_token_id"), list):
            raw["eos_token_id"] = raw["eos_token_id"][0]
        return cls(**_filter_known(cls, raw))


@dataclass
class Qwen3VLConfig:
    text_config: Qwen3VLTextConfig = field(default_factory=Qwen3VLTextConfig)
    vision_config: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Qwen3VLConfig:
        raw = dict(raw)
        text = Qwen3VLTextConfig.from_dict(raw.pop("text_config", {}))
        vision = Qwen3VLVisionConfig.from_dict(raw.pop("vision_config", {}))
        kept = _filter_known(cls, raw)
        kept.pop("text_config", None)
        kept.pop("vision_config", None)
        return cls(text_config=text, vision_config=vision, **kept)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> Qwen3VLConfig:
        cfg_path = Path(path)
        if cfg_path.is_dir():
            cfg_path = cfg_path / "config.json"
        with cfg_path.open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
