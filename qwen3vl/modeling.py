"""Standalone Qwen3-VL architecture.

A self-contained re-implementation of `Qwen3VLForConditionalGeneration` built on
plain `torch.nn`. Module and parameter names match the HuggingFace checkpoint
exactly, so `model.safetensors` loads with a strict `load_state_dict`.

Layout::

    Qwen3VLForConditionalGeneration
      model: Qwen3VLModel
        visual:         Qwen3VLVisionModel     (24 blocks + merger + 3 deepstack mergers)
        language_model: Qwen3VLTextModel       (28 decoder layers, GQA + QK-norm)
      lm_head: Linear                          (weight tied to embed_tokens)

Three details distinguish this from a vanilla VLM:

* **DeepStack** — the vision tower emits extra feature maps at blocks 5/11/17
  which are *added into* the hidden states of text decoder layers 0/1/2 at the
  image-token positions, rather than only being fed in at the embedding layer.
* **Interleaved M-RoPE** — text positions are 3-dimensional (time, height,
  width). The per-axis frequency bands are interleaved as THWTHW... instead of
  being laid out in contiguous chunks.
* **Bilinear position-embedding interpolation** — the vision tower holds a
  fixed 48x48 learned position table that is resampled to each image's patch
  grid on the fly, so arbitrary resolutions are supported.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig

ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
    "relu": F.relu,
}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: [a, b] -> [-b, a]."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query KV heads to full attention-head count."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class Qwen3VLRMSNorm(nn.Module):
    """Root-mean-square layer norm, computed in float32 regardless of input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# --------------------------------------------------------------------------- #
# vision tower
# --------------------------------------------------------------------------- #
class Qwen3VLVisionPatchEmbed(nn.Module):
    """Project flattened (C, T, P, P) pixel patches to the vision hidden size."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(target_dtype)).view(-1, self.embed_dim)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """2-D rotary frequencies; `dim` is half a head so (h, w) fill the head together."""

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.reset_buffers()

    def reset_buffers(self, device: torch.device | None = None) -> None:
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        # (num_tokens, 2) -> (num_tokens, 2 * dim/2)
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3VLVisionAttention(nn.Module):
    """Bidirectional attention over patches, restricted to one image at a time."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # (seq, heads, dim) -> (1, heads, seq, dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Each image is its own attention block; no token attends across images.
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = []
        for q, k, v in zip(
            torch.split(query_states, lengths, dim=2),
            torch.split(key_states, lengths, dim=2),
            torch.split(value_states, lengths, dim=2),
        ):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scaling)
            outputs.append(out.transpose(1, 2))  # -> (1, chunk, heads, dim)

        attn_output = torch.cat(outputs, dim=1).reshape(seq_length, -1)
        return self.proj(attn_output)


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionPatchMerger(nn.Module):
    """Fold each `spatial_merge_size**2` block of patches into one LLM token.

    `use_postshuffle_norm` selects whether the LayerNorm sees the pre-merge
    vision width (the output merger) or the post-merge concatenated width (the
    DeepStack mergers) — the checkpoint's norm shapes differ accordingly.
    """

    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_unit
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x)
        x = x.view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


def build_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Cumulative patch-count boundaries, one segment per frame."""
    counts = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cu_seqlens = counts.cumsum(dim=0, dtype=torch.int32)
    return F.pad(cu_seqlens, (1, 0), value=0)


def build_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """(row, col) index per patch, ordered to match the merge-block token layout."""
    device = grid_thw.device
    position_ids = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        m = spatial_merge_size

        hpos = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos = hpos.reshape(h // m, m, w // m, m).transpose(1, 2).flatten()

        wpos = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos = wpos.reshape(h // m, m, w // m, m).transpose(1, 2).flatten()

        position_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def build_bilinear_pos_embed_indices(
    grid_thw: torch.Tensor, num_grid_per_side: int, spatial_merge_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corner indices + weights that resample the 48x48 position table onto each grid.

    Returns ``(4, total_patches)`` index and weight tensors: one entry per
    bilinear corner, already permuted into merge-block token order.
    """
    side = num_grid_per_side
    m = spatial_merge_size
    device = grid_thw.device

    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)

        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)

        h_floor, w_floor = h_grid.int(), w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac, w_frac = h_grid - h_floor, w_grid - w_floor

        h_floor_off, h_ceil_off = h_floor * side, h_ceil * side
        corner_indices = [
            (h_floor_off[:, None] + w_floor[None, :]).flatten(),
            (h_floor_off[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_off[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_off[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]

        h_idx = torch.arange(h, device=device).view(h // m, m)
        w_idx = torch.arange(w, device=device).view(w // m, m)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)

        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])

    indices = torch.stack([torch.cat(p) for p in idx_parts])
    weights = torch.stack([torch.cat(p) for p in weight_parts])
    return indices, weights


class Qwen3VLVisionModel(nn.Module):
    """ViT over patch tokens, returning both merged LLM tokens and DeepStack features."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.num_grid_per_side = config.num_grid_per_side

        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.head_dim // 2)
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True) for _ in self.deepstack_visual_indexes]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Args:
            pixel_values: ``(total_patches, C * T * P * P)`` flattened patches.
            grid_thw: ``(num_images, 3)`` patch-grid extent per image.

        Returns:
            ``(merged_tokens, deepstack_features)`` — merged tokens are
            ``(num_llm_tokens, out_hidden_size)``, one DeepStack tensor of the
            same shape per configured index.
        """
        grid_thw = grid_thw.to(self.device)
        indices, weights = build_bilinear_pos_embed_indices(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size
        )
        position_ids = build_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = build_vision_cu_seqlens(grid_thw)

        hidden_states = self.patch_embed(pixel_values.to(self.device))
        pos_embeds = (self.pos_embed(indices) * weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = self.rotary_pos_emb(position_ids).reshape(hidden_states.shape[0], -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_features: list[torch.Tensor] = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)]
                deepstack_features.append(merger(hidden_states))

        return self.merger(hidden_states), deepstack_features


# --------------------------------------------------------------------------- #
# text decoder
# --------------------------------------------------------------------------- #
class Qwen3VLTextRotaryEmbedding(nn.Module):
    """Interleaved 3-D M-RoPE.

    Frequency band `i` is driven by axis `i % 3` (time/height/width) for the
    first `3 * min(section)` bands, and by time alone for the tail. That keeps
    neighbouring bands at adjacent wavelengths instead of splitting the spectrum
    into three contiguous chunks.
    """

    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3VLTextConfig) -> None:
        super().__init__()
        self.config = config
        self.mrope_section = list(config.mrope_section)
        self.reset_buffers()

    def reset_buffers(self, device: torch.device | None = None) -> None:
        head_dim = self.config.head_dim
        exponent = torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim
        inv_freq = 1.0 / (self.config.rope_theta**exponent)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: Sequence[int]) -> torch.Tensor:
        """Fold the (3, ...) per-axis frequencies into one interleaved band set."""
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):  # height, width
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: any tensor supplying the target dtype/device.
            position_ids: ``(3, batch, seq)`` time/height/width positions, or
                ``(batch, seq)`` which is broadcast across all three axes.
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, -1, -1)

        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        inv_freq = inv_freq.to(position_ids.device)
        pos = position_ids[:, :, None, :].float()

        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq @ pos).transpose(2, 3)  # (3, bs, seq, head_dim/2)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLTextAttention(nn.Module):
    """Grouped-query attention with per-head RMSNorm on Q and K."""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.scaling = self.head_dim**-0.5

        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)
        # Normalised over the head dim only, so no reshape is needed afterwards.
        self.q_norm = Qwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: "KVCache | None" = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            is_causal=attention_mask is None and query_states.shape[2] > 1,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        return self.o_proj(attn_output)


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: "KVCache | None" = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask, past_key_values)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class KVCache:
    """Per-layer key/value cache that grows by concatenation."""

    def __init__(self, num_layers: int) -> None:
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self) -> int:
        if self.key_cache[0] is None:
            return 0
        return self.key_cache[0].shape[2]

    def reset(self) -> None:
        self.key_cache = [None] * len(self.key_cache)
        self.value_cache = [None] * len(self.value_cache)


def build_causal_mask(
    attention_mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor | None:
    """Additive ``(bs, 1, q_len, kv_len)`` mask, or None when plain causal suffices."""
    if attention_mask is None and query_length == key_length:
        return None  # SDPA's built-in is_causal path
    if attention_mask is None and query_length == 1:
        return None  # single query attends to everything cached

    min_value = torch.finfo(dtype).min
    # Query i (absolute position key_length - query_length + i) may see keys <= that.
    causal = torch.arange(key_length, device=device)[None, :] > (
        torch.arange(query_length, device=device)[:, None] + (key_length - query_length)
    )
    mask = torch.zeros(query_length, key_length, dtype=dtype, device=device).masked_fill(causal, min_value)
    mask = mask[None, None, :, :].expand(batch_size, 1, -1, -1).clone()

    if attention_mask is not None:
        padding = attention_mask[:, None, None, :].to(device)
        mask = mask.masked_fill(padding == 0, min_value)
    return mask


class Qwen3VLTextModel(nn.Module):
    """Qwen3 decoder stack. Not text-only: DeepStack injects vision features here."""

    def __init__(self, config: Qwen3VLTextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)

    @staticmethod
    def _deepstack_process(
        hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ) -> torch.Tensor:
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :] + visual_embeds
        return hidden_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = inputs_embeds.shape
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        causal_mask = build_causal_mask(
            attention_mask,
            query_length=seq_length,
            key_length=past_length + seq_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            batch_size=batch_size,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_embeddings, causal_mask, past_key_values)
            # DeepStack: fuse vision features into the first len(embeds) layers.
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

        return self.norm(hidden_states)


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
class Qwen3VLModel(nn.Module):
    """Vision tower + text decoder, with M-RoPE index construction."""

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)
        self.rope_deltas: torch.Tensor | None = None

    # -- M-RoPE index construction ----------------------------------------- #
    @staticmethod
    def get_vision_position_ids(
        start_position: int,
        grid_thw: Sequence[int] | torch.Tensor,
        spatial_merge_size: int = 1,
        temp_merge_size: int = 1,
        time_interval: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """3-D (time, height, width) positions for one image or video's tokens."""
        t = int(grid_thw[0]) // temp_merge_size
        h = int(grid_thw[1]) // spatial_merge_size
        w = int(grid_thw[2]) // spatial_merge_size

        position_temporal = torch.arange(t, device=device) * time_interval
        position_width = torch.arange(w, device=device) + start_position
        position_height = torch.arange(h, device=device) + start_position

        position_width = position_width.repeat(h * t)
        position_height = position_height.repeat_interleave(w).repeat(t)
        position_temporal = position_temporal.repeat_interleave(h * w) + start_position
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def build_mm_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """0 for text, 1 for image placeholders, 2 for video placeholders."""
        token_types = torch.zeros_like(input_ids, dtype=torch.int32)
        token_types[input_ids == self.config.image_token_id] = 1
        token_types[input_ids == self.config.video_token_id] = 2
        return token_types

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(3, bs, seq)`` M-RoPE positions and the per-sample rope delta.

        Text runs advance all three axes together. An image consumes only
        ``max(h, w) // merge`` positions, so its tokens share a compact 2-D
        block of the position space instead of one index per token.
        """
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw = video_grid_thw.clone()
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        device = input_ids.device
        position_ids = torch.zeros(3, *input_ids.shape, dtype=torch.long, device=device)
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        rope_deltas = []
        for batch_idx in range(input_ids.shape[0]):
            current_input_ids = input_ids[batch_idx]
            token_types = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                keep = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[keep]
                token_types = token_types[keep]

            # Group the sequence into maximal runs of one modality.
            runs: list[tuple[int, int, int]] = []
            types = token_types.tolist()
            start = 0
            for i in range(1, len(types) + 1):
                if i == len(types) or types[i] != types[start]:
                    runs.append((types[start], start, i))
                    start = i

            current_pos = 0
            chunks = []
            for modality, start_idx, end_idx in runs:
                if modality == 0:
                    text_len = end_idx - start_idx
                    chunks.append(
                        torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality])
                    chunks.append(
                        self.get_vision_position_ids(
                            current_pos, grid_thw, spatial_merge_size=spatial_merge_size, device=device
                        )
                    )
                    current_pos += max(int(grid_thw[1]), int(grid_thw[2])) // spatial_merge_size

            llm_positions = torch.cat(chunks, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions
            rope_deltas.append(int(llm_positions.max()) + 1 - current_input_ids.shape[0])

        return position_ids, torch.tensor(rope_deltas, device=device).unsqueeze(1)

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None

        if input_ids is not None and has_multimodal and past_length == 0:
            if mm_token_type_ids is None:
                mm_token_type_ids = self.build_mm_token_type_ids(input_ids)
            position_ids, self.rope_deltas = self.get_rope_index(
                input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            return position_ids

        batch_size, seq_length, _ = inputs_embeds.shape
        device = inputs_embeds.device
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(device)
        else:
            position_ids = torch.arange(past_length, past_length + seq_length, device=device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)

        if self.rope_deltas is not None and past_length > 0:
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device)
        return position_ids

    # -- feature extraction -------------------------------------------------- #
    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        pixel_values = pixel_values.to(self.visual.device, self.visual.dtype)
        return self.visual(pixel_values, image_grid_thw)

    def get_placeholder_mask(self, input_ids: torch.Tensor, token_id: int, num_features: int) -> torch.Tensor:
        mask = input_ids == token_id
        num_placeholders = int(mask.sum())
        if num_placeholders != num_features:
            raise ValueError(
                f"placeholder/feature mismatch for token {token_id}: "
                f"{num_placeholders} placeholders vs {num_features} features"
            )
        return mask

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if input_ids is None and (pixel_values is not None or pixel_values_videos is not None):
            raise ValueError("input_ids is required to locate image/video placeholder positions")

        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        image_mask = video_mask = None
        deepstack_image_embeds = deepstack_video_embeds = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self.get_placeholder_mask(
                input_ids, self.config.image_token_id, image_embeds.shape[0]
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
            )

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_image_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = self.get_placeholder_mask(
                input_ids, self.config.video_token_id, video_embeds.shape[0]
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.unsqueeze(-1).expand_as(inputs_embeds), video_embeds
            )

        visual_pos_masks, deepstack_visual_embeds = self._merge_deepstack(
            image_mask, video_mask, deepstack_image_embeds, deepstack_video_embeds
        )

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    @staticmethod
    def _merge_deepstack(
        image_mask: torch.Tensor | None,
        video_mask: torch.Tensor | None,
        deepstack_image_embeds: list[torch.Tensor] | None,
        deepstack_video_embeds: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        """Interleave image and video DeepStack features onto a shared token mask."""
        if image_mask is not None and video_mask is not None:
            visual_pos_masks = image_mask | video_mask
            image_joint = image_mask[visual_pos_masks]
            video_joint = video_mask[visual_pos_masks]
            merged = []
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                joint = img_embed.new_zeros(int(visual_pos_masks.sum()), img_embed.shape[-1])
                joint[image_joint, :] = img_embed
                joint[video_joint, :] = vid_embed
                merged.append(joint)
            return visual_pos_masks, merged
        if image_mask is not None:
            return image_mask, deepstack_image_embeds
        if video_mask is not None:
            return video_mask, deepstack_video_embeds
        return None, None


class Qwen3VLForConditionalGeneration(nn.Module):
    """Qwen3-VL with the LM head. Weight names match the released checkpoint."""

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lm_head.weight.dtype

    def num_parameters(self) -> int:
        """Count distinct parameter tensors (tied weights counted once)."""
        seen = {id(p): p.numel() for p in self.parameters()}
        return sum(seen.values())

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> torch.Tensor:
        """Returns logits of shape ``(bs, seq_or_logits_to_keep, vocab_size)``."""
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        if logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        return self.lm_head(hidden_states)

    def new_cache(self) -> KVCache:
        return KVCache(self.config.text_config.num_hidden_layers)
