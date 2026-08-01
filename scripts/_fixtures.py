"""Deterministic prompt fixtures shared by every parity and benchmark script.

Both the torch reference and every compiled engine are measured on exactly these
inputs, so numbers stay comparable across phases. Images are generated from
fixed arithmetic rather than files — no binary assets in the repo, and the
inputs are reproducible on any machine.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen3-VL-2B-Instruct"
BASELINE_DIR = Path(__file__).resolve().parent.parent / "docs" / "baselines"


def solid_red(size: tuple[int, int] = (448, 448)) -> Image.Image:
    return Image.new("RGB", size, (200, 30, 30))


def gradient(size: tuple[int, int] = (640, 360)) -> Image.Image:
    img = Image.new("RGB", size)
    px = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            px[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img


def checkerboard(size: tuple[int, int] = (300, 200), cell: int = 25) -> Image.Image:
    img = Image.new("RGB", size)
    px = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            px[x, y] = (255, 255, 255) if (x // cell + y // cell) % 2 == 0 else (0, 0, 0)
    return img


def prompt_cases() -> list[tuple[str, list[dict]]]:
    """Named message lists covering text-only, one image, and two images."""
    return [
        (
            "text_only",
            [{"role": "user", "content": [{"type": "text", "text": "Name three prime numbers."}]}],
        ),
        (
            "single_image",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": solid_red()},
                        {"type": "text", "text": "What color dominates this image?"},
                    ],
                }
            ],
        ),
        (
            "two_images",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": solid_red()},
                        {"type": "image", "image": checkerboard()},
                        {"type": "text", "text": "Describe the difference between these two images."},
                    ],
                }
            ],
        ),
    ]


def vision_shapes() -> list[tuple[str, Image.Image]]:
    """Images chosen to span a useful range of patch counts for benchmarking."""
    return [
        ("small_256px", solid_red((256, 256))),
        ("medium_448px", solid_red((448, 448))),
        ("wide_640x360", gradient((640, 360))),
        ("large_896px", gradient((896, 896))),
        ("xlarge_1280x720", gradient((1280, 720))),
    ]
