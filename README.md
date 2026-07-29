# Tensor-RT-exp

Qwen3-VL-2B-Instruct rebuilt from scratch and pushed toward maximum-efficiency
serving: a standalone PyTorch implementation (no `transformers` at runtime),
then a split deployment — vision encoder compiled with TensorRT, text decoder
served with TensorRT-LLM — with custom Triton/CUDA/TileLang kernels along the
way. A learning project in inference optimization and GPU kernels.

## Layout

| Path | Contents |
|---|---|
| `qwen3vl/` | Standalone Qwen3-VL implementation: config, modeling, weight loading, preprocessing, generation. See its [README](qwen3vl/README.md) for architecture notes (DeepStack, interleaved M-RoPE, patch ordering) |
| `scripts/` | `download_qwen3vl.py` (fetch weights), `smoke_test.py`, `parity_check.py` (fp32 logits vs HF), `parity_generate.py` (greedy generation vs HF) |
| `docs/` | [Experiment plan](docs/trt-split-experiment-plan.md) (TRT split, validation gates) and [kernel roadmap](docs/kernel-learning-roadmap.md) (K1–K9 learning track) |
| `models/` | Weights, not committed — restore with the download script |

## Status

- [x] Standalone implementation, parity-verified vs `transformers` 5.10.2:
  vision tower bit-exact, logits 100% argmax agreement (fp32, 2-image prompt),
  greedy generation token-for-token identical
- [ ] Baselines frozen (Gate 0 of the experiment plan)
- [ ] Vision encoder → TensorRT engine
- [ ] Decoder → TensorRT-LLM (WSL2)
- [ ] Custom kernels (Triton → CUDA → TileLang)

## Setup

```bash
python scripts/download_qwen3vl.py   # ~4.3 GB into models/Qwen3-VL-2B-Instruct
python scripts/smoke_test.py         # load + generate on a test image
python scripts/parity_check.py       # numerical parity vs transformers
```

Needs torch ≥ 2.12 with CUDA, `safetensors`, `tokenizers`, `pillow`, `jinja2`;
parity scripts additionally need `transformers` ≥ 5.10. Hardware used: RTX 3060
12 GB (SM 8.6).

## Quick use

```python
import torch
from qwen3vl import GenerationConfig, Qwen3VLProcessor, generate, load_qwen3vl

model = load_qwen3vl("models/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16, device="cuda")
processor = Qwen3VLProcessor.from_pretrained("models/Qwen3-VL-2B-Instruct")

messages = [{"role": "user", "content": [
    {"type": "image", "image": "photo.jpg"},
    {"type": "text", "text": "What is in this image?"},
]}]
batch = processor.apply_chat_template(messages, device="cuda")
out = generate(model, **batch, config=GenerationConfig(max_new_tokens=128))
print(processor.decode(out.generated[0]))
```
