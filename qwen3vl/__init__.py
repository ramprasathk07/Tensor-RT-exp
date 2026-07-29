"""Standalone Qwen3-VL: architecture, checkpoint loading, preprocessing, decoding.

    from qwen3vl import load_qwen3vl, Qwen3VLProcessor, generate

    model = load_qwen3vl("models/Qwen3-VL-2B-Instruct", device="cuda")
    processor = Qwen3VLProcessor.from_pretrained("models/Qwen3-VL-2B-Instruct")
"""

from .config import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from .generation import GenerationConfig, GenerationOutput, generate
from .loading import load_qwen3vl, load_state_dict
from .modeling import (
    KVCache,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
from .processing import Qwen3VLImageProcessor, Qwen3VLProcessor, smart_resize

__all__ = [
    "GenerationConfig",
    "GenerationOutput",
    "KVCache",
    "Qwen3VLConfig",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLImageProcessor",
    "Qwen3VLModel",
    "Qwen3VLProcessor",
    "Qwen3VLTextConfig",
    "Qwen3VLTextModel",
    "Qwen3VLVisionConfig",
    "Qwen3VLVisionModel",
    "generate",
    "load_qwen3vl",
    "load_state_dict",
    "smart_resize",
]
