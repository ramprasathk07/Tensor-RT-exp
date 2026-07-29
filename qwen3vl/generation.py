"""Incremental decoding for the standalone Qwen3-VL model.

Prefill runs the full prompt (image tokens included) through the model and
fills the KV cache. Each decode step then feeds a single token back in.

Position handling is the part worth reading: the prompt's M-RoPE positions are
not ``0..seq_len-1`` because an image block consumes only ``max(h, w) // merge``
positions no matter how many tokens it expands to. Generated tokens therefore
continue from ``max(prompt_positions) + 1``, not from ``seq_len``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .modeling import Qwen3VLForConditionalGeneration


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    eos_token_ids: tuple[int, ...] = (151645, 151643)
    pad_token_id: int = 151643

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, **overrides) -> GenerationConfig:
        path = Path(model_dir) / "generation_config.json"
        raw = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)

        eos = raw.get("eos_token_id", [151645, 151643])
        kwargs = {
            "do_sample": raw.get("do_sample", False),
            "temperature": raw.get("temperature", 1.0),
            "top_p": raw.get("top_p", 1.0),
            "top_k": raw.get("top_k", 0),
            "repetition_penalty": raw.get("repetition_penalty", 1.0),
            "eos_token_ids": tuple(eos) if isinstance(eos, list) else (eos,),
            "pad_token_id": raw.get("pad_token_id", 151643),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    score = torch.gather(logits, 1, generated)
    # Divide positive scores, multiply negative ones, so both move downward.
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(1, generated, score)


def sample_next_token(logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    """Pick one token per batch row from ``(bs, vocab)`` logits."""
    if not config.do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / max(config.temperature, 1e-5)

    if config.top_k > 0:
        k = min(config.top_k, logits.shape[-1])
        threshold = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < config.top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        # Drop tokens whose predecessors already covered top_p, keeping the first.
        remove = probs.cumsum(dim=-1) - probs > config.top_p
        logits = logits.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)


@dataclass
class GenerationOutput:
    sequences: torch.Tensor
    """``(bs, prompt_len + generated_len)`` full token ids."""
    generated: torch.Tensor
    """``(bs, generated_len)`` newly produced token ids."""
    finished: torch.Tensor
    """``(bs,)`` bool — whether the row hit an EOS token."""


@torch.inference_mode()
def generate(
    model: Qwen3VLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    config: GenerationConfig | None = None,
) -> GenerationOutput:
    """Greedy or sampled decoding with a growing KV cache."""
    config = config or GenerationConfig()
    device = model.device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    batch_size = input_ids.shape[0]
    cache = model.new_cache()

    # -- prefill ------------------------------------------------------------ #
    prompt_positions = model.model.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=model.model.language_model.embed_tokens(input_ids),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        past_key_values=cache,
    )
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=prompt_positions,
        past_key_values=cache,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        logits_to_keep=1,
    )[:, -1, :]

    # Image blocks compress the position space, so continue from the max used.
    next_position = prompt_positions.amax(dim=(0, 2)) + 1  # (bs,)

    generated = torch.empty(batch_size, 0, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    eos_ids = torch.tensor(config.eos_token_ids, device=device)

    for _ in range(config.max_new_tokens):
        step_logits = apply_repetition_penalty(
            logits.float(), torch.cat([input_ids, generated], dim=1), config.repetition_penalty
        )
        next_token = sample_next_token(step_logits, config)
        next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, config.pad_token_id), next_token)

        generated = torch.cat([generated, next_token], dim=1)
        finished |= (next_token == eos_ids).any(dim=-1)
        if bool(finished.all()):
            break

        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)], dim=1
            )

        step_positions = next_position.view(1, batch_size, 1).expand(3, -1, -1)
        logits = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=step_positions,
            past_key_values=cache,
        )[:, -1, :]
        next_position = next_position + 1

    return GenerationOutput(
        sequences=torch.cat([input_ids, generated], dim=1),
        generated=generated,
        finished=finished,
    )
