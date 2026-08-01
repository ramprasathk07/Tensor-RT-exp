# Qwen3-VL-2B Split Deployment: Vision → TensorRT, Decoder → TensorRT-LLM

Experiment plan for compiling the vision encoder of Qwen3-VL-2B-Instruct into a
TensorRT engine and the text decoder into TensorRT-LLM, with validation gates at
every phase. Written for this machine and this repo.

---

## 1. Starting point (what already exists)

| Asset | Location | Status |
|---|---|---|
| Weights | `models/Qwen3-VL-2B-Instruct/` | 4.27 GB safetensors, 625 tensors |
| Standalone implementation | `qwen3vl/` | Pure torch, no `transformers` at runtime |
| Logit parity vs HF 5.10.2 | `tests/test_hf_parity.py` | PASS — vision bit-exact, logits max_abs 4.5e-4, argmax 100% |
| Generation parity | `tests/test_hf_generation.py` | PASS — token-for-token on 3 prompt shapes |
| Smoke test | `tests/smoke.py` | 3.7 s warm load, correct output |

The standalone implementation is the *reference oracle* for every later stage.
Every engine build gets compared against it, not against feelings.

### Hardware / platform constraints (verified)

| Item | Value | Consequence |
|---|---|---|
| GPU | RTX 3060, 12 GB, SM 8.6 (Ampere) | No FP8 (needs SM 8.9+). Quantization = INT8 KV / W4A16 AWQ. 2B in bf16 ≈ 4.3 GB weights — fits with room for KV cache |
| OS | Windows 10 Pro | TensorRT (vision) runs native Windows. **TensorRT-LLM has no Windows wheels** — decoder work happens in WSL2 |
| WSL2 | Ubuntu 24.04 installed, WSL 2 default | CUDA-in-WSL2 works with the existing Windows driver; do NOT install a Linux display driver inside WSL, only the CUDA toolkit |
| Python venv (Windows) | `F:\gitRepos\IssueFix-RL\.venv` | torch 2.12.0+cu126, transformers 5.10.2. WSL2 side needs its own venv |

---

## 2. Why this split is the right architecture

- **Vision encode is compute-bound**: one big batched pass over patch tokens,
  GEMM-heavy, no KV cache. TensorRT's layer fusion + tactic search is built for
  exactly this shape of workload.
- **Decode is memory-bound**: one token per step, performance = how fast you can
  stream weights and KV cache. TensorRT-LLM brings paged KV cache, inflight
  (continuous) batching, fused decode-attention kernels, quantized KV.
- Two different optimization regimes → two different tools. This is the standard
  production VLM serving pattern (TRT-LLM's own multimodal examples run the
  vision tower as a plain TRT engine and hand embeddings to the LLM engine).

## 3. The three Qwen3-VL-specific traps

These are the reasons a stock tutorial won't work unmodified. All three are
implemented and documented in `qwen3vl/` — read those files before the port.

### 3.1 DeepStack breaks the clean encoder/decoder interface
Vision blocks 5/11/17 each feed a separate patch merger; those three feature
maps are **added into the hidden states of decoder layers 0/1/2** at image-token
positions (`qwen3vl/modeling.py`, `_deepstack_process`). Consequences:

- The handoff between engines is not one embeddings tensor — it is **four**:
  merged tokens + 3 deepstack tensors, plus the visual-position mask.
- The decoder engine needs those as extra runtime inputs, with a scatter-add
  after layers 0, 1, 2. TRT-LLM's *PyTorch backend* makes this tractable (model
  definition is Python over their optimized ops). The legacy engine-building
  flow would require graph surgery — avoid it.

### 3.2 Interleaved M-RoPE
Positions are 3-D (t, h, w), sections [24, 20, 20], bands interleaved
`THWTHW…` with a 4-band pure-time tail (`apply_interleaved_mrope`). TRT-LLM has
mrope for Qwen2-VL (chunked layout); the interleaved variant may not exist in
its kernels. Also: position *allocation* is nonstandard — an image advances the
position counter by `max(h, w) // merge_size`, not by its token count, and
generated tokens continue from `max(prompt_positions) + 1`, not `seq_len`
(`qwen3vl/generation.py`). Any serving loop that assumes `position = seq_len`
produces silently wrong output after an image.

### 3.3 Variable-length blocked vision attention
Vision attention is bidirectional but blocked per image via `cu_seqlens` — no
token attends across image boundaries. The current implementation loops over
images with separate SDPA calls. That loop does not ONNX-export cleanly; it
must become either a block-diagonal additive mask (export-friendly) or a varlen
flash-attention kernel (Phase 3 of the kernel roadmap).

---

## 4. Pre-flight validation (do ALL of this before touching TRT)

Gate 0. Nothing below starts until every box here is checked.

### 4.1 Reference freeze
- [ ] Re-run `tests/test_hf_parity.py` and `tests/test_hf_generation.py`; record
      outputs into `docs/baselines/` as text files. These numbers are the
      contract every engine must meet.
- [ ] Pin exact versions in a `docs/baselines/environment.txt`: torch, CUDA,
      driver, transformers, GPU.
- [ ] Save reference tensors to disk for offline comparison
      (`torch.save`): for 2–3 fixed prompts, dump `pixel_values`,
      `image_grid_thw`, `input_ids`, merged vision tokens, all 3 deepstack
      tensors, final logits. Engines get diffed against these files without
      needing HF loaded.

### 4.2 Performance baseline (need numbers to know if TRT wins)
- [ ] Vision tower: latency vs patch count (e.g. 256 / 784 / 1664 / 4096
      patches), bf16, CUDA-graph-free, `torch.cuda.Event` timing, 50 iters
      after 10 warmup.
- [ ] Decoder prefill: tok/s at seq len 256 / 1024 / 4096.
- [ ] Decoder decode: tok/s single stream, batch 1 / 4 / 8, with KV cache.
- [ ] `torch.compile` variants of both — the honest baseline TRT must beat.
- [ ] One nsys trace of end-to-end generate; identify top-10 kernels. This is
      the map for the kernel roadmap.
- [ ] Record peak VRAM per configuration (12 GB budget is real).

### 4.3 Toolchain validation
- [ ] Windows: install TensorRT 10.x GA + matching CUDA. Verify with
      `trtexec --onnx=<any small onnx>` on a toy model (export a 2-layer MLP).
- [ ] WSL2: `nvidia-smi` works inside Ubuntu; install CUDA toolkit (WSL repo),
      fresh Python venv, `pip install tensorrt_llm` (or build from source if
      wheel/CUDA mismatch). Run their smallest text-LLM quickstart end to end.
- [ ] WSL2: run TRT-LLM's Qwen3 (text-only) example — proves the Qwen family
      path works before adding vision complications.
- [ ] Check TRT-LLM support matrix for Qwen3-VL. If supported: the port is
      mostly reading + wiring. If not: the port is writing a model definition
      in their PyTorch backend using `qwen3vl/modeling.py` as the spec.
- [ ] Disk: engines + ONNX + duplicated WSL2 checkpoints ≈ 15–20 GB. Verify free
      space (F: has ~160 GB — fine, but check C: if WSL default vhd lives there).

### 4.4 Numerical tolerance policy (decide now, not after)
- Vision engine (bf16/fp16): cosine ≥ 0.999 per merged token, max_abs ≤ 2e-2
  vs the fp32 reference dumps; argmax of downstream logits must not change on
  the frozen prompts.
- Decoder engine: greedy generation must be token-identical on the 3 frozen
  prompts for ≥ 32 tokens. Where TRT-LLM sampling differs, compare logits at
  each step instead (max_abs ≤ 5e-2 in bf16 is realistic; argmax agreement
  ≥ 99.9%).
- Any gate failure → bisect by dumping per-module outputs (vision tower is
  already bit-exact in torch, so any drift is engine-side).

---

## 5. Phase 1 — Vision encoder → TensorRT (native Windows)

**Goal**: single TRT engine `vision.plan` taking
`(pixel_values, pos_embed_indices, pos_embed_weights, rope_cos, rope_sin, block_mask)`
→ `(merged_tokens, deepstack_0, deepstack_1, deepstack_2)`.

Steps:
1. **Export-friendly forward.** New `qwen3vl/export_vision.py` wrapping
   `Qwen3VLVisionModel`:
   - Replace the per-image SDPA loop with one SDPA call + additive
     block-diagonal mask built from `cu_seqlens` (mask is an engine *input*,
     built on host — keeps data-dependent logic out of the graph).
   - Precompute bilinear pos-embed indices/weights and rope cos/sin on host
     (already pure functions of `grid_thw`) and pass as inputs. The graph then
     contains only gather + GEMMs + norms — trivially exportable.
2. **Parity in torch** of the wrapper vs the original tower (must stay
   bit-exact — same ops, different packaging).
3. **ONNX export**, opset ≥ 17, dynamic axis = `num_patches` (and
   `num_merged_tokens` on outputs). Check with `onnxruntime` CPU first: fast
   iteration on export bugs before TRT enters the picture.
4. **Engine build**: `trtexec` with optimization profiles, e.g.
   min 256 / opt 1664 / max 6144 patches, `--bf16`. Save build log.
5. **Validation** against the frozen reference dumps (gate from §4.4).
6. **Benchmark** vs torch and torch.compile at the same patch counts. Inspect
   with `trtexec --dumpProfile` and TREx: which layers fused, which tactic won,
   where the time goes.

Deliverables: `vision.plan`, parity report, benchmark table, one-page notes on
what TRT fused and what it refused to fuse (input for the kernel roadmap).

## 6. Phase 2 — Decoder → TensorRT-LLM (WSL2)

**Goal**: TRT-LLM-served decoder consuming the four vision tensors, matching
frozen greedy outputs, beating torch decode tok/s.

Steps:
1. **Support-matrix decision** (from §4.3): native Qwen3-VL support → run it,
   validate, then read their deepstack/mrope wiring as the learning exercise.
   No support → port `qwen3vl/modeling.py` decoder into a TRT-LLM PyTorch
   backend model definition:
   - their paged attention op in place of SDPA + `KVCache`;
   - custom interleaved-mrope rotary (start with torch ops inside the model
     def — correctness first, kernel later);
   - deepstack scatter-add after layers 0/1/2, mask + 3 tensors as extra
     model inputs.
2. **Text-only parity first**: no images, greedy, 3 prompts, token-identical
   vs the frozen dumps. Isolates decoder correctness from the handoff.
3. **Handoff wiring**: vision `.plan` outputs → prompt-embedding table
   (merged tokens replace image-placeholder embeddings) + deepstack side
   inputs. Positions: port the `max(h,w)//merge` allocation and the
   `max(prompt_positions)+1` continuation rule — this is the most likely
   silent-failure point; add an assert comparing position tensors against the
   torch implementation for each frozen prompt.
4. **End-to-end parity** on frozen multimodal prompts (gate §4.4).
5. **Throughput**: single-stream decode tok/s, then inflight batching with
   4/8/16 concurrent requests, then INT8 KV cache. Record VRAM at each point.

Deliverables: engine/config, parity report, throughput table
(torch vs TRT-LLM × batch × quantization), notes on what their runtime does
that the naive loop doesn't (scheduling, KV paging, graph capture).

## 7. Phase 3 — Custom kernels

Separate document: `docs/kernel-learning-roadmap.md`. Entry criterion: Phases
0–1 done (need profiles to know which kernels matter, and a working TRT engine
to plug plugins into).

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| TRT-LLM has no Qwen3-VL and PyTorch-backend port stalls on deepstack | medium | Fallback experiment: decoder in TRT-LLM as *text-only* + deepstack disabled, quantify quality loss on image prompts; still teaches the full serving stack |
| ONNX export fights (bicubic-resize ops, data-dependent shapes) | medium | Preprocessing stays outside the engine (host-side, already separate in `qwen3vl/processing.py`); only the tower is exported |
| 12 GB VRAM ceiling with both engines + KV resident | medium | Vision engine can run fp16 (≈0.8 GB); INT8 KV; cap max batch; worst case run vision on CPU-TRT for functional tests |
| WSL2 perf noise in benchmarks | low | Compare WSL2-torch vs WSL2-TRT-LLM (same substrate), never Windows-torch vs WSL2-TRT-LLM |
| Version drift between the two environments | high | `environment.txt` per side, checked into `docs/baselines/` |

## 9. Success criteria

1. Vision TRT engine ≥ 1.5× faster than torch.compile tower at opt shape,
   parity gate passed.
2. Decoder TRT-LLM ≥ 1.5× single-stream decode tok/s vs torch loop, and
   ≥ 4× aggregate tok/s at batch 8 with inflight batching, parity gate passed.
3. End-to-end image→answer latency reported for 3 image sizes, with a
   breakdown (preprocess / vision / prefill / decode).
4. Written comparison: what each stack did to earn its speedup (fusion lists,
   kernel names from nsys before/after). This artifact is the learning output.
