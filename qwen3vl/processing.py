"""Input pipeline for Qwen3-VL: image patchification, chat template, tokenization.

Images are resized so the patch grid is a whole number of 2x2 merge blocks and
the total pixel count lands inside the configured budget, then flattened into
the ``(num_patches, C * T * P * P)`` layout the vision tower consumes. Patch
order is ``[frame][h-block][w-block][h-in-block][w-in-block]`` — the vision
tower's position-id and merge logic both assume exactly this ordering.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer

IMAGE_PAD_TOKEN = "<|image_pad|>"
VIDEO_PAD_TOKEN = "<|video_pad|>"


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio: int = 200,
) -> tuple[int, int]:
    """Round (h, w) to multiples of `factor` while keeping area within budget.

    Returns dimensions whose aspect ratio stays close to the original and whose
    product lies in ``[min_pixels, max_pixels]``.
    """
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"aspect ratio must stay under {max_ratio}, got {max(height, width) / min(height, width):.1f}"
        )

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Qwen3VLImageProcessor:
    """Resize, normalise and patchify images into vision-tower input."""

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 65536,
        max_pixels: int = 16777216,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean
        self.image_std = image_std

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Qwen3VLImageProcessor:
        path = Path(model_dir) / "preprocessor_config.json"
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        size = raw.get("size", {})
        return cls(
            patch_size=raw.get("patch_size", 16),
            temporal_patch_size=raw.get("temporal_patch_size", 2),
            merge_size=raw.get("merge_size", 2),
            min_pixels=size.get("shortest_edge", 65536),
            max_pixels=size.get("longest_edge", 16777216),
            image_mean=tuple(raw.get("image_mean", (0.5, 0.5, 0.5))),
            image_std=tuple(raw.get("image_std", (0.5, 0.5, 0.5))),
        )

    @property
    def resize_factor(self) -> int:
        """Grid must be whole merge blocks, so resize in patch_size*merge_size steps."""
        return self.patch_size * self.merge_size

    def _to_tensor(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        """Return a uint8 ``(C, H, W)`` tensor."""
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).expand(3, -1, -1)
            if tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            if tensor.dtype != torch.uint8:
                # Floats are assumed to be in [0, 1].
                tensor = (tensor.float() * 255.0).round().clamp(0, 255).to(torch.uint8)
            return tensor

        image = image.convert("RGB")
        array = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return array.view(image.size[1], image.size[0], 3).permute(2, 0, 1)

    def preprocess_one(self, image: Image.Image | torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Patchify a single image. Returns ``(patches, (t, h, w))``."""
        tensor = self._to_tensor(image)
        _, height, width = tensor.shape

        resized_h, resized_w = smart_resize(
            height, width, self.resize_factor, self.min_pixels, self.max_pixels
        )
        # The reference resizes while still uint8, so the round-trip through
        # integers is part of the expected output — do it before rescaling.
        tensor = F.interpolate(
            tensor.float().unsqueeze(0), size=(resized_h, resized_w), mode="bicubic", antialias=True
        ).squeeze(0)
        tensor = tensor.round().clamp(0, 255).to(torch.uint8)

        tensor = tensor.float() / 255.0
        mean = torch.tensor(self.image_mean).view(3, 1, 1)
        std = torch.tensor(self.image_std).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # A still image is one frame repeated to fill the temporal patch.
        frames = tensor.unsqueeze(0).repeat(self.temporal_patch_size, 1, 1, 1)

        grid_t = frames.shape[0] // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        m, p = self.merge_size, self.patch_size

        patches = frames.reshape(grid_t, self.temporal_patch_size, 3, grid_h // m, m, p, grid_w // m, m, p)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = patches.reshape(grid_t * grid_h * grid_w, 3 * self.temporal_patch_size * p * p)
        return flat.contiguous(), (grid_t, grid_h, grid_w)

    def __call__(self, images: list[Image.Image | torch.Tensor]) -> dict[str, torch.Tensor]:
        all_patches, all_grids = [], []
        for image in images:
            patches, grid = self.preprocess_one(image)
            all_patches.append(patches)
            all_grids.append(grid)
        return {
            "pixel_values": torch.cat(all_patches, dim=0),
            "image_grid_thw": torch.tensor(all_grids, dtype=torch.long),
        }


class Qwen3VLProcessor:
    """Chat template + tokenizer + image processor, producing model kwargs."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: Qwen3VLImageProcessor,
        chat_template: str,
        config: dict[str, Any],
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.chat_template = chat_template
        self.config = config
        self.merge_unit = image_processor.merge_size**2

        from jinja2 import Environment

        env = Environment(trim_blocks=True, lstrip_blocks=True)
        env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        self._template = env.from_string(chat_template)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Qwen3VLProcessor:
        model_dir = Path(model_dir)
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

        with (model_dir / "chat_template.json").open("r", encoding="utf-8") as fh:
            chat_template = json.load(fh)["chat_template"]
        with (model_dir / "config.json").open("r", encoding="utf-8") as fh:
            config = json.load(fh)

        return cls(tokenizer, Qwen3VLImageProcessor.from_pretrained(model_dir), chat_template, config)

    def token_id(self, token: str) -> int:
        tid = self.tokenizer.token_to_id(token)
        if tid is None:
            raise KeyError(f"token {token!r} not in vocabulary")
        return tid

    @staticmethod
    def collect_images(messages: list[dict[str, Any]]) -> list[Image.Image | torch.Tensor]:
        """Pull every image object out of the message content blocks, in order."""
        images: list[Image.Image | torch.Tensor] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    image = block.get("image", block.get("url", block.get("path")))
                    if isinstance(image, (str, Path)):
                        image = Image.open(image)
                    if image is None:
                        raise ValueError(f"image block has no loadable source: {block}")
                    images.append(image)
        return images

    def render(self, messages: list[dict[str, Any]], add_generation_prompt: bool = True) -> str:
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            add_vision_id=False,
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Render, expand image placeholders, tokenize. Returns model kwargs."""
        text = self.render(messages, add_generation_prompt=add_generation_prompt)
        images = self.collect_images(messages)

        batch: dict[str, torch.Tensor] = {}
        if images:
            vision_inputs = self.image_processor(images)
            batch["pixel_values"] = vision_inputs["pixel_values"]
            batch["image_grid_thw"] = vision_inputs["image_grid_thw"]

            # One placeholder per merged token, so text length matches feature count.
            for grid in vision_inputs["image_grid_thw"].tolist():
                num_tokens = (grid[0] * grid[1] * grid[2]) // self.merge_unit
                text = text.replace(IMAGE_PAD_TOKEN, "<|placeholder|>" * num_tokens, 1)
            text = text.replace("<|placeholder|>", IMAGE_PAD_TOKEN)

        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        batch["input_ids"] = torch.tensor([encoding.ids], dtype=torch.long)
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])

        return {k: v.to(device) for k, v in batch.items()}

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
