"""Load the downloaded Qwen3-VL-2B-Instruct snapshot and run one image+text prompt."""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen3-VL-2B-Instruct"


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    image = Image.new("RGB", (256, 256), (200, 30, 30))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What color is this image? Answer in one word."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    reply = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]

    print(f"params: {model.num_parameters() / 1e9:.2f}B  dtype: {model.dtype}")
    print(f"reply: {reply.strip()}")


if __name__ == "__main__":
    main()
