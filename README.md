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
- [x] Baselines frozen — fp32 reference tensors, 3-D M-RoPE positions and
  greedy generations captured as the oracle for every later runtime
- [x] Vision encoder → TensorRT engine: **1.6–2.2× faster** than torch eager,
  and closer to the fp32 model than the bf16 pipeline it replaces
  ([results](docs/phase1-results.md))
- [ ] Decoder → TensorRT-LLM (WSL2)
- [ ] Custom kernels (Triton → CUDA → TileLang)

### Findings so far

- TensorRT 11 removed the FP16/BF16 builder flags — networks are strongly
  typed, so precision must come from the exported ONNX. bf16 additionally needs
  the `torch.export` path, which also produced a 44% smaller graph.
- **bf16 is the wrong precision for this vision tower** despite being the
  checkpoint dtype. Activations stay bounded, so bf16 spends mantissa on range
  it never uses: 3.4× further from fp32 than fp16, and 25% slower.
- Measure against fp32, not against the bf16 pipeline. Scored that way, TRT
  fp16 reproduces fp32 output on 3/3 fixtures where bf16 manages 2/3.
- 86% of engine time is GEMMs; TensorRT already fuses LayerNorm and residuals
  into them. Custom-kernel headroom is in the decoder, not the vision tower.
- `torch.compile` cannot be benchmarked on Windows (Triton has no Windows
  wheels), so Triton kernel work belongs in WSL2.

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
