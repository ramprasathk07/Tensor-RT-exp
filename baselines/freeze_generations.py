"""Freeze fp32 greedy generations as a cross-runtime oracle.

The tensor dumps from `gate0_freeze.py` are the right oracle for anything that
can load torch tensors. TensorRT-LLM runs in a different environment (WSL2,
different venv, different torch), so it needs something portable: plain token
ids and text in JSON.

Greedy decoding with a fixed prompt is deterministic, so any runtime claiming to
serve this model should reproduce these token sequences exactly — or explain
why not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures import BASELINE_DIR, MODEL_DIR, prompt_cases

from qwen3vl import GenerationConfig, Qwen3VLProcessor, generate, load_qwen3vl

MAX_NEW_TOKENS = 64
OUT_PATH = BASELINE_DIR / "reference_generations.json"


def main() -> None:
    dtype = torch.float32
    print(f"loading reference model ({dtype}) ...")
    model = load_qwen3vl(MODEL_DIR, dtype=dtype, device="cuda")
    processor = Qwen3VLProcessor.from_pretrained(MODEL_DIR)
    config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    record = {
        "dtype": "float32",
        "max_new_tokens": MAX_NEW_TOKENS,
        "sampling": "greedy",
        "model": MODEL_DIR.name,
        "cases": {},
    }

    for name, messages in prompt_cases():
        batch = processor.apply_chat_template(messages, device="cuda")
        out = generate(model, **batch, config=config)
        tokens = out.generated[0].cpu().tolist()
        text = processor.decode(out.generated[0])

        # The text prompt is portable across runtimes; images are regenerated
        # from `_fixtures.py`, which builds them from fixed arithmetic.
        record["cases"][name] = {
            "prompt_text": processor.render(messages),
            "prompt_token_count": int(batch["input_ids"].shape[1]),
            "num_images": len(processor.collect_images(messages)),
            "image_grid_thw": batch["image_grid_thw"].cpu().tolist()
            if "image_grid_thw" in batch else [],
            "generated_token_ids": tokens,
            "generated_text": text,
        }
        print(f"  {name:<14} {len(tokens):>3} tokens  {text.strip()[:70]!r}")

    OUT_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
