# GPU Kernel Learning Roadmap — Qwen3-VL as the Vehicle

A beginner-to-advanced kernel-writing track using this repo's Qwen3-VL-2B
pipeline as the testbed. Every kernel here replaces a real op in
`qwen3vl/modeling.py`, has a built-in correctness oracle (the parity-verified
torch implementation), and a real baseline to beat. That combination — real
model, exact reference, honest benchmark — is what makes this portfolio-grade
rather than tutorial-grade.

Target hardware: RTX 3060, SM 8.6 (Ampere), 12 GB. Relevant limits: 48 SMs,
no FP8, no TMA (Hopper-only), 100 KB usable shared memory per SM, ~360 GB/s
memory bandwidth. Memory-bound kernels top out at bandwidth; know that number
— "% of 360 GB/s achieved" is the honest score for every decode-side kernel.

---

## 0. Ground rules (apply to every kernel)

1. **Correctness before speed.** Every kernel ships with a pytest that compares
   against the eager `qwen3vl` op over: multiple shapes (including ragged /
   non-power-of-2), bf16 + fp32, and a fixed-seed random sweep. Tolerance
   explicit and justified.
2. **Benchmark discipline.** `triton.testing.do_bench` (or CUDA events, 50
   iters / 10 warmup), report: time, achieved GB/s, achieved TFLOPs, vs
   baseline (cuBLAS / SDPA / flash-attn / TRT-LLM kernel). A losing kernel is a
   valid result *if the write-up explains why it loses* (occupancy? bank
   conflicts? no async copy?). Profile with Nsight Compute (`ncu`): occupancy,
   memory throughput, warp-stall reasons.
3. **One markdown write-up per kernel** in `docs/kernels/`: the math, the
   tiling diagram, what was tried, ncu screenshots, final numbers. These
   write-ups ARE the portfolio.
4. **Integration target order**: (a) drop into the torch pipeline via
   `torch.library.custom_op` — fastest iteration; (b) call from the TRT-LLM
   PyTorch-backend model definition — near-zero extra work once (a) works;
   (c) wrap as a TensorRT `IPluginV3` plugin — C++, hardest, do once, late.

### Toolchain ladder

| Tool | When | Why |
|---|---|---|
| **Triton** | Kernels 1–6 | Python-level, autotuner, removes indexing pain; industry-standard for exactly this class of kernel |
| **CUDA C++** | Re-do kernels 2 and 5 after Triton versions work | Forces understanding of what Triton hid: shared-memory layout, `cp.async`, bank conflicts, warp shuffles |
| **TileLang** | Kernels 6–7 | Tile-level DSL; sweet spot is decode-attention / quant-GEMM style kernels; newer skill, differentiates a profile |

A note on the naming: **FlashMLA does not apply here.** FlashMLA is DeepSeek's
decode kernel for *MLA* (multi-head latent attention, compressed KV). Qwen3-VL
uses *GQA* (16 query heads over 8 KV heads). The equivalent skill in this repo
is a flash-*decoding* GQA kernel (kernel 6) — same family: memory-bound,
single-query attention, split-KV parallelism. Understanding kernel 6 makes the
FlashMLA paper readable; that is the right order.

---

## Phase A — Foundations (week-scale, no model code yet)

Skip nothing here; everything later is these primitives composed.

- **A1. Vector add → fused elementwise.** Triton tutorial 1. Learn: grid,
  program_id, masking, why memory-bound kernels are all about coalescing.
- **A2. Softmax (row-wise, online).** Learn: row-per-program tiling, numerical
  max-subtraction, the online-rescale trick — this is 50% of flash attention
  already.
- **A3. Matmul with shared-memory tiling + autotune.** Learn: block tiling,
  `tl.dot`, tensor cores via bf16 inputs, autotune configs, why K-loop order
  matters. Score vs cuBLAS at the model's real shapes: (784×1024)·(1024×3072),
  (seq×2048)·(2048×6144).
- **A4. RMSNorm forward.** First real-model kernel, directly reusable: the
  exact op in `Qwen3VLRMSNorm` (fp32 accumulation inside, bf16 out). Validate
  against it. Trivial, but now the harness (pytest + bench + write-up) exists.

Reading alongside: PMPP (Kirk & Hwu) ch. 1–6; Triton tutorials 1–8;
"How to think about GPU performance" style roofline material. Learn to read
ncu's speed-of-light section before Phase B.

## Phase B — Model kernels, memory-bound tier (the highest-value learning)

### Kernel 1 — Fused RMSNorm + residual add
`hidden = residual + attn_out` then `RMSNorm(hidden)` — two passes over the
tensor in eager mode, one in the kernel. Learn: kernel fusion arithmetic (bytes
saved vs bytes moved), in-place hazards.
**Oracle**: decoder layer pre/post tensors. **Win condition**: ≥ 1.6× vs the
two-op eager sequence at (1, 2048) and (8, 2048).

### Kernel 2 — Fused QK-RMSNorm + interleaved M-RoPE  ⭐ signature kernel
Qwen3-VL applies per-head RMSNorm to Q and K, then interleaved 3-axis rope
(`apply_interleaved_mrope` + `apply_rotary_pos_emb` in `modeling.py`). Eager
cost: two norms + strided-slice frequency interleave + rotate-half — a pile of
small launches per decode step. Fuse into one kernel: load q/k head, normalize
in fp32, apply precomputed cos/sin, write.

This kernel is *distinctive*: interleaved M-RoPE is new enough that no standard
library ships it fused. It is simultaneously beginner-shaped (elementwise per
head, no cross-thread reduction beyond the norm) and genuinely useful — it goes
straight into the TRT-LLM PyTorch-backend port from the experiment plan.
**Oracle**: `Qwen3VLTextAttention` intermediates. **Win condition**: measurable
decode-step improvement in the torch pipeline (whole-step timing, batch 1 and 8),
plus a clean latency win vs the unfused op sequence in isolation.

### Kernel 3 — Fused SwiGLU
`down(silu(gate(x)) * up(x))`: fuse the silu-multiply epilogue into the
gate/up GEMM (Triton matmul with epilogue), keep `down` as cuBLAS. Learn:
GEMM epilogue fusion, why "fuse everything" is wrong (the down GEMM wants
cuBLAS's tactics).
**Oracle**: `Qwen3VLTextMLP`. **Win condition**: beat eager
(3 GEMMs + 2 elementwise) end-to-end; losing to torch.compile is acceptable if
the write-up shows *what* torch.compile generated (inspect with
`TORCH_LOGS=output_code`).

## Phase C — Attention tier

### Kernel 4 — Varlen bidirectional flash-attention (vision)  ⭐ feeds Phase-1 TRT work
The vision tower needs blocked attention over ragged per-image segments
(`cu_seqlens`). Implement flash-attention-2-style: Q-block outer loop, KV-block
inner loop, online softmax, segment boundaries from `cu_seqlens` (like
flash-attn's varlen API). No causal mask — simpler than the LLM case; 16 heads
× head_dim 64 — small tiles, fits Ampere shared memory easily.
Learn: the full flash-attention algorithm, ragged indexing, why varlen beats
padding (compute scales with real tokens).
**Oracle**: vision tower per-block outputs (bit-exact torch reference exists).
**Win condition**: ≥ SDPA-with-block-mask throughput at 1664 and 4096 patches;
this becomes the candidate for the TensorRT `IPluginV3` exercise.

### Kernel 5 — CUDA C++ re-implementation of kernel 4 (or 2)
Same spec, raw CUDA: `cp.async` double-buffered shared-memory pipeline, tensor
core `mma` via `wmma` or inline PTX, swizzled shared-memory layout to kill bank
conflicts. Expect this to take longer than everything before it combined; that
is the point. Compare instruction-level behaviour against the Triton version
in ncu.
Learn: what Triton's compiler was doing for free.

### Kernel 6 — GQA flash-decoding kernel (the FlashMLA-family skill)
Single query token, long KV: parallelism must come from the KV axis. Implement
split-KV: partition KV into chunks, each program computes partial
softmax-weighted sums + running max/denominator, second small kernel (or
atomic-free reduction pass) merges partials. Handle GQA (2 query heads share
each KV head → load KV once per group).
This is decode-attention in every modern serving stack (flash-decoding,
PagedAttention v2's kernel, FlashMLA). Do it in Triton first, then **TileLang**
— TileLang's abstractions map naturally onto this kernel and having the same
kernel in both is an excellent write-up.
**Oracle**: decoder attention at decode step with cache lengths 512–32k.
**Win condition**: beat naive SDPA-over-full-cache at ≥ 4k KV length; report
% of 360 GB/s achieved (target ≥ 60%).

## Phase D — Advanced / integration tier (optional, after the TRT experiment)

- **Kernel 7 — W4A16 dequant-GEMM** (AWQ-style): pack int4 weights, dequant in
  registers inside the K-loop, tensor-core accumulate. Compare vs TRT-LLM's own
  W4A16 kernels. TileLang candidate. This is the quantization-kernel skill.
- **Kernel 8 — TensorRT `IPluginV3` plugin** wrapping kernel 4 into the Phase-1
  vision engine. C++ plumbing: serialization, shape inference, workspace,
  format negotiation. One-time cost, high leverage on a CV.
- **Kernel 9 — Fused patch-merger** (LayerNorm → 2×2 shuffle → GEMM → GELU →
  GEMM, `Qwen3VLVisionPatchMerger`): only worth it after TREx shows TRT failed
  to fuse it; otherwise skip — measure first.

---

## Suggested sequence and pacing

```
A1→A4 (foundations)                    ~1-2 weeks
K1, K2 (fused norm, mrope)             ~1-2 weeks   ← first portfolio pieces
K4 (varlen flash-attn, Triton)         ~2-3 weeks   ← centerpiece
K3 (swiglu)                            ~1 week, can interleave
K6 (flash-decoding, Triton+TileLang)   ~2-3 weeks   ← second centerpiece
K5 (CUDA rewrite)                      ~2-4 weeks, start whenever blocked elsewhere
K7/K8/K9                               after the TRT split experiment
```

Interleave with the experiment plan: K2 and K6 plug directly into the TRT-LLM
PyTorch-backend decoder; K4 plugs into the vision path and later the TRT plugin.
Kernels with a deployment destination read very differently from kernels in a
vacuum.

## Portfolio packaging (do as you go, not at the end)

- Repo layout: `kernels/<name>/{kernel.py, kernel.cu, test_.py, bench.py, README.md}`.
- Each README: problem, tiling diagram (ASCII fine), bench table vs baselines,
  ncu findings, what failed. Failure analysis is the strongest signal of
  actual understanding — keep the dead ends in the write-up.
- One top-level post/README: "Optimizing Qwen3-VL-2B inference from scratch" —
  links the standalone implementation (already done, parity-proven), the TRT
  split experiment, and the kernel series. That narrative arc — model →
  serving → kernels — is exactly the inference-engineer skill story.
- Numbers to headline: end-to-end tok/s progression (eager → compile → TRT-LLM
  → +custom kernels), and per-kernel % of hardware roofline achieved.

## Reference list

- Programming Massively Parallel Processors (Kirk & Hwu) — ch. 1–6 before
  Phase B, ch. on memory/performance before Phase C.
- Triton official tutorials (01–11) — A-phase companion.
- FlashAttention-2 paper — before K4. Flash-Decoding blog post — before K6.
- FlashMLA repo / DeepSeek papers — *after* K6, as the "now you can read this"
  payoff.
- TileLang docs + examples (their MLA/GEMM samples) — before K6's TileLang pass.
- Nsight Compute docs: speed-of-light + warp-stall sections.
- TRT-LLM PyTorch-backend model definitions — read their attention wrapper
  before integrating K2/K6.
