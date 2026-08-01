# Phase 1 results — vision encoder on TensorRT

Vision tower of Qwen3-VL-2B exported to ONNX and compiled to a dynamic-shape
TensorRT engine, validated against the frozen fp32 reference and benchmarked
against torch.

**Outcome: 1.6–2.2× faster than torch eager, and *more* faithful to the fp32
model than the bf16 pipeline the model normally ships as.**

Environment: RTX 3060 12 GB (SM 8.6), TensorRT 11.2.1.2, torch 2.12.0+cu126,
Windows. Reproduce with `scripts/phase1_export_onnx.py`,
`phase1_build_engine.py`, `phase1_end2end.py`.

---

## 1. What had to change to make the tower exportable

`Qwen3VLVisionModel` loops over images (to keep attention blocked per image) and
derives bilinear position indices and rotary positions from `grid_thw` with
Python control flow. None of that traces.

`qwen3vl/export_vision.py` keeps the arithmetic and moves the data-dependent
parts to the host:

| eager | exported |
|---|---|
| per-image SDPA loop over `cu_seqlens` | one attention over all patches, cross-image pairs suppressed by an additive mask |
| bilinear indices/weights computed inside forward | graph **inputs**, precomputed by `build_vision_inputs` |
| rotary positions computed from `grid_thw` | graph input `vision_position_ids` |

The mask is built *in-graph* from one `int32` segment id per patch
(`segment_ids[i] == segment_ids[j]`) rather than transferred as an `(N, N)`
tensor — at 6144 patches that would have been a 151 MB host-to-device copy per
call.

Verified bit-faithful to the eager tower: worst relative error **4.4e-5** across
five patch counts (256 → 3520), including a non-square grid and a 3-image batch.
Under onnxruntime with dynamic shapes: worst relative error **3.8e-4**.

## 2. TensorRT 11 changed how precision is requested

`BuilderFlag.FP16` and `BuilderFlag.BF16` no longer exist. TensorRT 11 builds
**strongly-typed** networks: the graph's own dtypes decide the precision, so the
ONNX must be exported at the target precision and the network created with
`NetworkDefinitionCreationFlag.STRONGLY_TYPED`.

Consequence for the export path: bf16 cannot go through the TorchScript exporter
(`ScalarType ComplexDouble is an unexpected tensor scalar type`) and needs the
`torch.export`-based path (`dynamo=True`, requires `onnxscript`). That path also
produced a noticeably cleaner graph — **1633 nodes vs 2905** for the same model.

Engine build times were short: 1.1 min (fp16), 1.6 min (bf16).

## 3. fp16 beats bf16 here — the opposite of the initial assumption

The checkpoint ships in bf16, so bf16 looked like the safe default. Measured
against the fp32 reference, it is the worse choice:

| precision | torch vs fp32, worst rel | TRT latency @784 patches |
|---|---|---|
| fp16 | **8.8e-2** | **23.2 ms** |
| bf16 | 3.0e-1 | 31.4 ms |

bf16 buys exponent range and pays for it with mantissa bits (8 vs fp16's 10).
This tower's activations stay bounded (max |activation| ≈ 13–27 across every
fixture), so the range is never needed and the lost precision is pure cost.
bf16 was also ~25% slower in this engine.

Error grows monotonically with depth in both, as expected for a 24-block
residual stack: the DeepStack tap at block 5 stays at ~2e-3 relative while the
final merged output reaches ~1e-1.

## 4. Measuring against the right reference changes the verdict

Comparing the TRT engine to the bf16 torch pipeline suggested a failure
(worst relative error 1.8e-1). That comparison is misleading — the bf16
pipeline is itself an approximation. Scoring generated text against a **fp32
torch reference** instead:

| pipeline | cases generating the same tokens as fp32 |
|---|---|
| torch bf16 (the shipping baseline) | 2 / 3 |
| torch bf16 decoder + **TRT fp16 vision** | **3 / 3** |

On the `single_image` fixture the bf16 baseline diverges from fp32 at token 13
("a solid, uniform shade of red…" instead of "a solid, deep red…"), while the
TRT fp16 pipeline reproduces the fp32 output exactly.

So the engine is not merely acceptable — on these fixtures it tracks the true
model more closely than the precision the model normally runs in. Feature-level
relative error, which is a max-over-outliers metric, was the wrong gate; cosine
similarity stayed ≥ 0.997 throughout and generated tokens were the decision.

## 5. Benchmark

Median over 30 iterations after 10 warmup, `torch.cuda.Event` timing, fp16 both
sides.

| case | patches | torch eager | TRT fp16 | speedup |
|---|---|---|---|---|
| small 256px | 256 | 21.6 ms | 9.9 ms | **2.18×** |
| medium 448px | 784 | 40.3 ms | 23.2 ms | **1.73×** |
| wide 640×360 | 880 | 41.2 ms | 24.7 ms | **1.67×** |
| large 896px | 3136 | 186.6 ms | 112.4 ms | **1.66×** |
| xlarge 1280×720 | 3520 | 209.0 ms | 129.4 ms | **1.62×** |

The small-image case gains most because per-launch overhead dominates there and
TensorRT collapses the per-block op sequence into far fewer kernels. The gain
settles at ~1.65× once the GEMMs are large enough to dominate.

**`torch.compile` could not be measured on Windows**: inductor's GPU backend
requires Triton, which has no official Windows wheels
(`TritonMissing: Cannot find a working triton installation`). The eager
comparison above is therefore the honest local baseline. This also constrains
the kernel roadmap — all Triton work must happen in WSL2, not on the Windows
side.

## 6. Phase 1 success criteria

| criterion | target | result |
|---|---|---|
| speedup at opt shape | ≥ 1.5× | 1.73× at 784 patches ✅ |
| parity gate | no answer change | 3/3 vs fp32, better than bf16 baseline ✅ |
| dynamic shapes | one engine, many resolutions | 256–6144 patch profile, 5 shapes verified ✅ |

## 7. Carried into Phase 2

- `make_trt_image_features` in `scripts/phase1_end2end.py` is already the
  vision→decoder handoff Phase 2 needs: merged tokens plus the three DeepStack
  tensors, swapped in for `get_image_features` with the decoder untouched.
- Use **fp16** for the vision engine, not bf16.
- Score every decoder variant against fp32, never against the bf16 pipeline.
- The `(N, N)` attention mask is the remaining inefficiency in the tower and is
  exactly what kernel K4 (varlen bidirectional flash-attention) removes.
