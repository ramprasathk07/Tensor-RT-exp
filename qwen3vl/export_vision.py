"""Export-friendly repackaging of the Qwen3-VL vision tower.

`Qwen3VLVisionModel` is written for eager PyTorch: it loops over images to keep
attention blocked per image, and it derives bilinear position-embedding indices
and rotary positions from `grid_thw` with Python-level control flow. None of
that traces into ONNX.

This module keeps the arithmetic identical but moves every data-dependent
computation to the host:

* bilinear indices/weights and 2-D rotary positions become **inputs**, computed
  once per request by `build_vision_inputs` (pure functions of `grid_thw`);
* the per-image attention loop becomes a single fused attention over all
  patches, with cross-image attention suppressed by a segment-id mask built
  inside the graph — cheap to transfer (one int per patch) and cheap to build.

The result is a graph of gathers, GEMMs, norms and one attention per block.
`ExportableVisionTower` is weight-identical to the eager tower: it holds the
same module, so it loads the same checkpoint and produces the same numbers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling import (
    Qwen3VLVisionModel,
    apply_rotary_pos_emb_vision,
    build_bilinear_pos_embed_indices,
    build_vision_position_ids,
)


def build_vision_inputs(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Precompute every `grid_thw`-derived tensor the exported graph needs.

    Args:
        grid_thw: ``(num_images, 3)`` patch-grid extent per image.
        num_grid_per_side: side of the learned position table (48 for this model).
        spatial_merge_size: 2 for this model.

    Returns dict of graph inputs, all on `device`:
        ``pos_indices`` ``(4, N)`` int32, ``pos_weights`` ``(4, N)`` float32,
        ``vision_position_ids`` ``(N, 2)`` int32, ``segment_ids`` ``(N,)`` int32.
    """
    grid_thw = grid_thw.to("cpu")
    indices, weights = build_bilinear_pos_embed_indices(
        grid_thw, num_grid_per_side, spatial_merge_size
    )
    position_ids = build_vision_position_ids(grid_thw, spatial_merge_size)

    # One segment per frame; attention never crosses a segment boundary.
    counts = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    segment_ids = torch.repeat_interleave(
        torch.arange(counts.shape[0], dtype=torch.int32), counts
    )

    return {
        "pos_indices": indices.to(torch.int32).to(device),
        "pos_weights": weights.to(torch.float32).to(device),
        "vision_position_ids": position_ids.to(torch.int32).to(device),
        "segment_ids": segment_ids.to(device),
    }


class ExportableVisionTower(nn.Module):
    """Traceable wrapper over `Qwen3VLVisionModel` with static control flow."""

    def __init__(self, tower: Qwen3VLVisionModel) -> None:
        super().__init__()
        self.tower = tower
        self.config = tower.config
        self.deepstack_visual_indexes = list(tower.deepstack_visual_indexes)

    def _attention(
        self,
        block_attn,
        hidden_states: torch.Tensor,
        attn_bias: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """One vision attention over all patches at once, masked per segment."""
        seq_length = hidden_states.shape[0]
        qkv = block_attn.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, block_attn.num_heads, block_attn.head_dim)
        query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)

        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # (seq, heads, dim) -> (1, heads, seq, dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attn_bias, scale=block_attn.scaling
        )
        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1)
        return block_attn.proj(attn_output)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pos_indices: torch.Tensor,
        pos_weights: torch.Tensor,
        vision_position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            pixel_values: ``(N, C*T*P*P)`` flattened patches.
            pos_indices: ``(4, N)`` bilinear corner indices into the position table.
            pos_weights: ``(4, N)`` bilinear corner weights.
            vision_position_ids: ``(N, 2)`` (row, col) per patch.
            segment_ids: ``(N,)`` image/frame id per patch.

        Returns:
            ``(merged, deepstack_0, deepstack_1, deepstack_2)``.
        """
        tower = self.tower

        hidden_states = tower.patch_embed(pixel_values)

        pos_embeds = (tower.pos_embed(pos_indices.long()) * pos_weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary = tower.rotary_pos_emb(vision_position_ids.float())
        rotary = rotary.reshape(hidden_states.shape[0], -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        # Additive bias: 0 within an image, -inf across images. Built in-graph
        # from one int per patch instead of transferring an (N, N) mask.
        same_segment = segment_ids.unsqueeze(1) == segment_ids.unsqueeze(0)
        attn_bias = torch.zeros(
            same_segment.shape, dtype=hidden_states.dtype, device=hidden_states.device
        ).masked_fill(~same_segment, torch.finfo(hidden_states.dtype).min)
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)

        deepstack_outputs: list[torch.Tensor] = []
        for layer_num, block in enumerate(tower.blocks):
            hidden_states = hidden_states + self._attention(
                block.attn, block.norm1(hidden_states), attn_bias, cos, sin
            )
            hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))

            if layer_num in self.deepstack_visual_indexes:
                merger = tower.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)]
                deepstack_outputs.append(merger(hidden_states))

        merged = tower.merger(hidden_states)
        return merged, deepstack_outputs[0], deepstack_outputs[1], deepstack_outputs[2]


def input_names() -> list[str]:
    return ["pixel_values", "pos_indices", "pos_weights", "vision_position_ids", "segment_ids"]


def output_names() -> list[str]:
    return ["merged", "deepstack_0", "deepstack_1", "deepstack_2"]


def dynamic_axes() -> dict[str, dict[int, str]]:
    """Patch count varies per image; merged-token count is patches / merge_unit."""
    return {
        "pixel_values": {0: "num_patches"},
        "pos_indices": {1: "num_patches"},
        "pos_weights": {1: "num_patches"},
        "vision_position_ids": {0: "num_patches"},
        "segment_ids": {0: "num_patches"},
        "merged": {0: "num_tokens"},
        "deepstack_0": {0: "num_tokens"},
        "deepstack_1": {0: "num_tokens"},
        "deepstack_2": {0: "num_tokens"},
    }
