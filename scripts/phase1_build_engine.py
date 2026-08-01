"""Phase 1b: build the TensorRT vision engine, validate it, benchmark it.

Builds a dynamic-shape engine from `artifacts/vision_tower.onnx`, checks its
outputs against the fp32 torch wrapper, and times it against torch eager and
torch.compile at several patch counts.

    python scripts/phase1_build_engine.py            # fp16 (default)
    python scripts/phase1_build_engine.py --bf16
    python scripts/phase1_build_engine.py --rebuild
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _fixtures import BASELINE_DIR, MODEL_DIR, vision_shapes

from qwen3vl import Qwen3VLImageProcessor, load_qwen3vl
from qwen3vl.export_vision import (
    ExportableVisionTower,
    build_vision_inputs,
    input_names,
    output_names,
)

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Optimization profile: covers 256x256 up to ~1280x720 images with headroom.
MIN_PATCHES, OPT_PATCHES, MAX_PATCHES = 256, 1024, 6144
MERGE_UNIT = 4
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}


def requested_precision() -> str:
    for p in ("fp32", "bf16", "fp16"):
        if f"--{p}" in sys.argv:
            return p
    return "fp16"


def engine_path(precision: str) -> Path:
    return ARTIFACT_DIR / f"vision_tower_{precision}.plan"


def build_engine(precision: str) -> Path:
    out_path = engine_path(precision)
    if out_path.exists() and "--rebuild" not in sys.argv:
        print(f"=== reusing {out_path.name} ({out_path.stat().st_size / 1e6:.0f} MB) ===")
        print("    pass --rebuild to force\n")
        return out_path

    onnx_file = ARTIFACT_DIR / f"vision_tower_{precision}.onnx"
    if not onnx_file.exists():
        raise SystemExit(f"missing {onnx_file.name} — run phase1_export_onnx.py --{precision}")

    print(f"=== building TensorRT engine ({precision}) ===")
    print(f"    TensorRT {trt.__version__}, profile {MIN_PATCHES}/{OPT_PATCHES}/{MAX_PATCHES} patches")

    builder = trt.Builder(TRT_LOGGER)
    # TensorRT 11 dropped the FP16/BF16 builder flags: networks are strongly
    # typed, so precision is whatever the ONNX graph declares.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with onnx_file.open("rb") as fh:
        if not parser.parse(fh.read()):
            for i in range(parser.num_errors):
                print(f"  parser error: {parser.get_error(i)}")
            raise SystemExit("ONNX parse failed")
    print(f"    parsed: {network.num_layers} layers, "
          f"{network.num_inputs} inputs, {network.num_outputs} outputs")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)

    profile = builder.create_optimization_profile()
    shapes = {
        "pixel_values": lambda n: (n, 1536),
        "pos_indices": lambda n: (4, n),
        "pos_weights": lambda n: (4, n),
        "vision_position_ids": lambda n: (n, 2),
        "segment_ids": lambda n: (n,),
    }
    for name, shape_fn in shapes.items():
        profile.set_shape(name, shape_fn(MIN_PATCHES), shape_fn(OPT_PATCHES), shape_fn(MAX_PATCHES))
    config.add_optimization_profile(profile)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")
    build_time = time.time() - t0

    out_path.write_bytes(serialized)
    print(f"    built in {build_time / 60:.1f} min -> {out_path.name} "
          f"({out_path.stat().st_size / 1e6:.0f} MB)\n")
    return out_path


class TRTVisionRunner:
    """Thin execution wrapper: torch tensors in, torch tensors out."""

    def __init__(self, plan_path: Path) -> None:
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        num_patches = inputs["pixel_values"].shape[0]
        for name in input_names():
            self.context.set_input_shape(name, tuple(inputs[name].shape))
        if not self.context.all_shape_inputs_specified:
            raise RuntimeError("shape inputs unspecified")

        # Cast to whatever the engine declares, so callers can pass fp32 and let
        # an fp16 engine consume it without a separate preprocessing path.
        buffers = {}
        for name in input_names():
            want = TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            tensor = inputs[name].to(want).contiguous()
            buffers[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())

        outputs = []
        for name in output_names():
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            out = torch.empty(shape, dtype=dtype, device="cuda")
            outputs.append(out)
            self.context.set_tensor_address(name, out.data_ptr())

        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        del buffers
        return outputs


def make_inputs(processor, images, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    vision = processor(images)
    extra = build_vision_inputs(vision["image_grid_thw"], 48, 2, device="cuda")
    return {
        "pixel_values": vision["pixel_values"].to("cuda", dtype),
        "grid_thw": vision["image_grid_thw"].to("cuda"),
        **extra,
    }


def bench(fn, iters: int = 30, warmup: int = 8) -> float:
    """Median milliseconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    precision = requested_precision()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    plan = build_engine(precision)

    processor = Qwen3VLImageProcessor.from_pretrained(MODEL_DIR)
    runner = TRTVisionRunner(plan)

    print(f"=== parity: TRT {precision} vs torch fp32 ===")
    model = load_qwen3vl(MODEL_DIR, dtype=torch.float32, device="cuda")
    wrapper = ExportableVisionTower(model.model.visual).eval()

    rows = []
    worst = 0.0
    for name, image in vision_shapes():
        inputs = make_inputs(processor, [image], torch.float32)
        args = tuple(inputs[k] for k in input_names())
        with torch.inference_mode():
            ref = wrapper(*args)
        trt_out = runner(inputs)

        print(f"{name}  patches={inputs['pixel_values'].shape[0]}")
        for i, out_name in enumerate(output_names()):
            a, b = ref[i].float(), trt_out[i].float()
            diff = (a - b).abs()
            scale = a.abs().max().clamp(min=1e-6)
            rel = float(diff.max() / scale)
            cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
            worst = max(worst, rel)
            print(f"  {out_name:<14} max_abs={diff.max():.3e}  rel={rel:.3e}  cos={cos:.6f}")

    tol = 3e-2 if precision == "fp16" else 6e-2
    verdict = "PASS" if worst < tol else "FAIL"
    print(f"parity {verdict}  (worst rel {worst:.3e}, tol {tol})\n")

    print("=== benchmark ===")
    torch_fp16 = ExportableVisionTower(model.model.visual).eval().half()
    compiled = torch.compile(torch_fp16, dynamic=True)

    print(f"{'case':<18}{'patches':>8}{'torch fp16':>13}{'compile':>11}{'TRT':>11}{'speedup':>10}")
    for name, image in vision_shapes():
        inputs_h = make_inputs(processor, [image], torch.float16)
        inputs_f = make_inputs(processor, [image], torch.float32)
        args_h = tuple(inputs_h[k] for k in input_names())
        n = inputs_h["pixel_values"].shape[0]

        with torch.inference_mode():
            t_eager = bench(lambda: torch_fp16(*args_h))
            try:
                t_comp = bench(lambda: compiled(*args_h), iters=20, warmup=4)
            except Exception:
                t_comp = float("nan")
        t_trt = bench(lambda: runner(inputs_f))

        best_torch = min(t_eager, t_comp) if t_comp == t_comp else t_eager
        rows.append({"case": name, "patches": n, "torch_fp16_ms": round(t_eager, 3),
                     "compile_ms": round(t_comp, 3), "trt_ms": round(t_trt, 3),
                     "speedup_vs_best_torch": round(best_torch / t_trt, 2)})
        print(f"{name:<18}{n:>8}{t_eager:>12.2f}m{t_comp:>10.2f}m{t_trt:>10.2f}m"
              f"{best_torch / t_trt:>9.2f}x")

    out = {"precision": precision, "tensorrt": trt.__version__,
           "profile": [MIN_PATCHES, OPT_PATCHES, MAX_PATCHES],
           "parity_worst_rel": worst, "parity": verdict, "benchmark": rows}
    (BASELINE_DIR / f"phase1_vision_trt_{precision}.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {BASELINE_DIR / f'phase1_vision_trt_{precision}.json'}")

    del model, wrapper, torch_fp16, compiled
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
