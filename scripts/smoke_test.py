"""Quick check that the standalone model loads its weights and runs end to end."""

import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen3vl import GenerationConfig, Qwen3VLProcessor, generate, load_qwen3vl

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen3-VL-2B-Instruct"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    t0 = time.time()
    model = load_qwen3vl(MODEL_DIR, dtype=torch.bfloat16, device=device)
    print(f"loaded in {time.time() - t0:.1f}s | params {model.num_parameters() / 1e9:.3f}B")

    processor = Qwen3VLProcessor.from_pretrained(MODEL_DIR)

    image = Image.new("RGB", (448, 448), (200, 30, 30))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What color is this image? Answer in one word."},
            ],
        }
    ]

    batch = processor.apply_chat_template(messages, device=device)
    print({k: tuple(v.shape) for k, v in batch.items()})

    t0 = time.time()
    out = generate(model, **batch, config=GenerationConfig(max_new_tokens=32, do_sample=False))
    elapsed = time.time() - t0

    reply = processor.decode(out.generated[0])
    print(f"generated {out.generated.shape[1]} tokens in {elapsed:.2f}s")
    print(f"reply: {reply.strip()!r}")


if __name__ == "__main__":
    main()
