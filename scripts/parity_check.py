"""Verify the standalone implementation matches HuggingFace numerically.

Runs both models sequentially (freeing each before loading the next so a single
GPU suffices) on identical inputs and compares logits, argmax agreement, and
preprocessing output.
"""

import gc
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen3-VL-2B-Instruct"
DTYPE = torch.float32  # float32 so tolerances reflect implementation, not bf16 rounding
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_images() -> list[Image.Image]:
    """Two differently shaped images so smart_resize and grid handling both get exercised."""
    red = Image.new("RGB", (448, 448), (200, 30, 30))
    gradient = Image.new("RGB", (640, 360))
    pixels = gradient.load()
    for x in range(640):
        for y in range(360):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return [red, gradient]


def hf_inputs_and_logits(images):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": "Describe both images in detail."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=DTYPE, attn_implementation="sdpa"
    ).to(DEVICE)
    model.eval()

    inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**{k: v for k, v in inputs.items()}, use_cache=False)
    logits = out.logits.float().cpu()

    # Reference vision features, to localise any mismatch to tower vs decoder.
    with torch.inference_mode():
        vision_out = model.model.get_image_features(
            inputs["pixel_values"].to(DTYPE), inputs["image_grid_thw"], return_dict=True
        )
    vis_merged = torch.cat(vision_out.pooler_output, dim=0).float().cpu()
    vis_deepstack = [d.float().cpu() for d in vision_out.deepstack_features]

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in inputs.items()}, logits, vis_merged, vis_deepstack


def mine_logits(inputs, want_vision: bool):
    from qwen3vl import load_qwen3vl

    model = load_qwen3vl(MODEL_DIR, dtype=DTYPE, device=DEVICE)
    kwargs = {
        "input_ids": inputs["input_ids"].to(DEVICE),
        "attention_mask": inputs["attention_mask"].to(DEVICE),
        "pixel_values": inputs["pixel_values"].to(DEVICE),
        "image_grid_thw": inputs["image_grid_thw"].to(DEVICE),
    }
    if "mm_token_type_ids" in inputs:
        kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"].to(DEVICE)

    with torch.inference_mode():
        logits = model(**kwargs).float().cpu()

    vis_merged = vis_deepstack = None
    if want_vision:
        with torch.inference_mode():
            merged, deepstack = model.model.get_image_features(
                inputs["pixel_values"].to(DEVICE), inputs["image_grid_thw"].to(DEVICE)
            )
        vis_merged = merged.float().cpu()
        vis_deepstack = [d.float().cpu() for d in deepstack]

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return logits, vis_merged, vis_deepstack


def report(name: str, a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).abs()
    denom = a.abs().max().clamp(min=1e-6)
    print(
        f"{name:<28} max_abs={diff.max():.3e}  mean_abs={diff.mean():.3e}  "
        f"rel={diff.max() / denom:.3e}  shape={tuple(a.shape)}"
    )
    return float(diff.max())


def compare_processors(images) -> None:
    from transformers import AutoProcessor

    from qwen3vl import Qwen3VLProcessor

    hf = AutoProcessor.from_pretrained(MODEL_DIR)
    mine = Qwen3VLProcessor.from_pretrained(MODEL_DIR)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": "Describe both images in detail."},
            ],
        }
    ]
    hf_batch = hf.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    my_batch = mine.apply_chat_template(messages)

    print("\n--- preprocessing ---")
    same_ids = torch.equal(hf_batch["input_ids"], my_batch["input_ids"])
    print(f"input_ids identical:         {same_ids}  (hf {tuple(hf_batch['input_ids'].shape)}, "
          f"mine {tuple(my_batch['input_ids'].shape)})")
    print(f"image_grid_thw identical:    {torch.equal(hf_batch['image_grid_thw'], my_batch['image_grid_thw'])}"
          f"  {hf_batch['image_grid_thw'].tolist()}")
    report("pixel_values", hf_batch["pixel_values"].float(), my_batch["pixel_values"].float())


def main() -> None:
    images = build_images()

    print(f"device={DEVICE} dtype={DTYPE}\n--- loading reference (transformers) ---")
    inputs, hf_logits, hf_vis, hf_deep = hf_inputs_and_logits(images)
    print(f"prompt tokens: {inputs['input_ids'].shape[1]}, patches: {inputs['pixel_values'].shape[0]}")

    print("--- loading standalone implementation ---")
    my_logits, my_vis, my_deep = mine_logits(inputs, want_vision=True)

    print("\n--- vision tower ---")
    report("merged image tokens", hf_vis, my_vis)
    for i, (a, b) in enumerate(zip(hf_deep, my_deep)):
        report(f"deepstack[{i}]", a, b)

    print("\n--- logits ---")
    max_diff = report("full logits", hf_logits, my_logits)
    agree = (hf_logits.argmax(-1) == my_logits.argmax(-1)).float().mean()
    print(f"argmax agreement:            {agree * 100:.2f}%")

    last_hf, last_mine = hf_logits[0, -1], my_logits[0, -1]
    cos = torch.nn.functional.cosine_similarity(last_hf, last_mine, dim=0)
    print(f"last-position cosine:        {cos:.8f}")
    print(f"last-position top-1:         hf={last_hf.argmax().item()} mine={last_mine.argmax().item()}")

    compare_processors(images)

    tol = 2e-3
    print(f"\nPARITY {'PASS' if max_diff < tol and agree > 0.999 else 'FAIL'} (logit tol {tol})")


if __name__ == "__main__":
    main()
