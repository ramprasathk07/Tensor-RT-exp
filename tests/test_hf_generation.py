"""Compare greedy generation between HuggingFace and the standalone model.

Exercises the KV-cache decode path and M-RoPE position bookkeeping, which the
single-forward logits check in `parity_check.py` does not cover.
"""

import gc
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures import MODEL_DIR
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 48


def make_cases() -> list[tuple[str, list[dict]]]:
    red = Image.new("RGB", (448, 448), (200, 30, 30))
    checker = Image.new("RGB", (300, 200))
    pixels = checker.load()
    for x in range(300):
        for y in range(200):
            pixels[x, y] = (255, 255, 255) if (x // 25 + y // 25) % 2 == 0 else (0, 0, 0)

    return [
        (
            "text-only",
            [{"role": "user", "content": [{"type": "text", "text": "Name three prime numbers."}]}],
        ),
        (
            "single-image",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": red},
                        {"type": "text", "text": "What color dominates this image?"},
                    ],
                }
            ],
        ),
        (
            "two-images",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": red},
                        {"type": "image", "image": checker},
                        {"type": "text", "text": "Describe the difference between these two images."},
                    ],
                }
            ],
        ),
    ]


def run_hf(cases):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=DTYPE, attn_implementation="sdpa"
    ).to(DEVICE)
    model.eval()

    results = {}
    for name, messages in cases:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        results[name] = (new_tokens.cpu().tolist(), processor.decode(new_tokens, skip_special_tokens=True))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_mine(cases):
    from qwen3vl import GenerationConfig, Qwen3VLProcessor, generate, load_qwen3vl

    processor = Qwen3VLProcessor.from_pretrained(MODEL_DIR)
    model = load_qwen3vl(MODEL_DIR, dtype=DTYPE, device=DEVICE)
    config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    results = {}
    for name, messages in cases:
        batch = processor.apply_chat_template(messages, device=DEVICE)
        out = generate(model, **batch, config=config)
        tokens = out.generated[0].cpu().tolist()
        results[name] = (tokens, processor.decode(out.generated[0]))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main() -> None:
    cases = make_cases()
    print(f"device={DEVICE} dtype={DTYPE} max_new_tokens={MAX_NEW_TOKENS}")

    print("--- reference generation ---")
    hf = run_hf(cases)
    print("--- standalone generation ---")
    mine = run_mine(cases)

    all_match = True
    for name, _ in cases:
        hf_tokens, hf_text = hf[name]
        my_tokens, my_text = mine[name]

        # Trim trailing pad the standalone emits after EOS for finished rows.
        common = min(len(hf_tokens), len(my_tokens))
        prefix_match = hf_tokens[:common] == my_tokens[:common]
        divergence = next(
            (i for i in range(common) if hf_tokens[i] != my_tokens[i]), None
        )

        status = "MATCH" if prefix_match else f"DIVERGES at token {divergence}"
        all_match &= prefix_match
        print(f"\n[{name}] {status}  (hf {len(hf_tokens)} tok, mine {len(my_tokens)} tok)")
        print(f"  hf:   {hf_text.strip()!r}")
        print(f"  mine: {my_text.strip()!r}")

    print(f"\nGENERATION PARITY {'PASS' if all_match else 'FAIL'}")


if __name__ == "__main__":
    main()
