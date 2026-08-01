# Repository layout

Each directory is one stage of the pipeline. Anything that produces a number
lives in `tests/`; anything that produces an artifact lives in `export/` or
`engine/`.

```
qwen3vl/        the model itself — importable library, no experiment code
export/         torch  -> ONNX
engine/         ONNX   -> TensorRT engine, plus runtime/benchmark/profile
tests/          everything that validates or measures correctness
baselines/      freezes the fp32 reference oracle
tools/          one-off utilities (weight download)
docs/           plans, results, and generated reports under docs/baselines/
artifacts/      .onnx and .plan files (gitignored, regenerate from export/)
models/         weights (gitignored, restore with tools/download_weights.py)
```

## Files

| path | what it does |
|---|---|
| `paths.py` | canonical paths (`MODEL_DIR`, `ARTIFACT_DIR`, `engine_path()`, …) |
| `qwen3vl/modeling.py` | vision tower, text decoder, KV cache |
| `qwen3vl/export_vision.py` | traceable repackaging of the vision tower |
| `qwen3vl/{config,loading,processing,generation}.py` | config, weight loading, preprocessing, decoding |
| `export/vision_onnx.py` | validates the export wrapper, writes `vision_tower_<precision>.onnx` |
| `engine/build_vision.py` | ONNX → `.plan`, dynamic-shape optimization profile |
| `engine/runtime.py` | `TRTVisionRunner`, dtype mapping, timing and error helpers |
| `engine/benchmark_vision.py` | TRT vs torch vs torch.compile timings |
| `engine/profile_vision.py` | per-layer profile, op-family breakdown |
| `tests/fixtures.py` | the fixed images and prompts every measurement uses |
| `tests/smoke.py` | load the model, generate once |
| `tests/test_hf_parity.py` | standalone implementation vs `transformers` (logits) |
| `tests/test_hf_generation.py` | standalone implementation vs `transformers` (tokens) |
| `tests/test_engine_parity.py` | TensorRT vs torch at matched precision |
| `tests/test_end2end.py` | fp32 vs bf16 vs TRT pipelines, compared on generated text |
| `tests/reference_hf_generate.py` | run the stock `transformers` model, for sanity |
| `baselines/freeze_tensors.py` | fp32 tensors + 3-D M-RoPE positions → `docs/baselines/tensors/` |
| `baselines/freeze_generations.py` | fp32 greedy tokens → JSON, readable from any runtime |
| `tools/download_weights.py` | fetch the checkpoint |

## Imports

Entry points sit one level below the repo root, so running
`python tests/smoke.py` puts `tests/` on `sys.path` rather than the root. Each
entry point opens with:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
```

after which `qwen3vl`, `engine.runtime` and `tests.fixtures` all resolve no
matter where the script was launched from.

## Order of operations

```bash
python tools/download_weights.py                # weights
python tests/test_hf_parity.py                  # standalone == transformers
python baselines/freeze_tensors.py              # freeze the fp32 oracle
python baselines/freeze_generations.py

python export/vision_onnx.py --fp16             # torch -> ONNX
python engine/build_vision.py --fp16            # ONNX -> .plan
python tests/test_engine_parity.py --fp16       # numeric check
python tests/test_end2end.py --fp16             # does the answer change?
python engine/benchmark_vision.py --fp16        # speed
python engine/profile_vision.py --fp16          # where the time goes
```
