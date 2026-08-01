"""Phase 1b: build the TensorRT vision engine, validate it, benchmark it.

Builds a dynamic-shape engine from `artifacts/vision_tower.onnx`, checks its
outputs against the fp32 torch wrapper, and times it against torch eager and
torch.compile at several patch counts.

    python scripts/phase1_build_engine.py            # fp16 (default)
    python scripts/phase1_build_engine.py --bf16
    python scripts/phase1_build_engine.py --rebuild
"""

from __future__ import annotations

import copy
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
        for name in input_names():
            if not self.context.set_input_shape(name, tuple(inputs[name].shape)):
                raise RuntimeError(f"shape rejected for {name}: {tuple(inputs[name].shape)} "
                                   f"(outside optimization profile?)")

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

    # Three-way comparison. Comparing TRT against fp32 alone cannot tell whether
    # drift comes from TensorRT or simply from running in reduced precision, so
    # torch at the same precision is measured as the control.
    print(f"=== parity: fp32 reference vs torch {precision} vs TRT {precision} ===")
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    model = load_qwen3vl(MODEL_DIR, dtype=torch.float32, device="cuda")
    wrapper_fp32 = ExportableVisionTower(model.model.visual).eval()
    tower_low = copy.deepcopy(model.model.visual).to(torch_dtype).eval()
    wrapper_low = ExportableVisionTower(tower_low).eval()

    rows = []
    worst_trt_vs_torch = 0.0
    worst_torch_vs_fp32 = 0.0

    def rel_err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        a, b = a.float(), b.float()
        rel = float((a - b).abs().max() / a.abs().max().clamp(min=1e-6))
        cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        return rel, cos

    for name, image in vision_shapes():
        inputs = make_inputs(processor, [image], torch.float32)
        args32 = tuple(inputs[k] for k in input_names())
        args_low = tuple(
            inputs[k].to(torch_dtype) if inputs[k].is_floating_point() else inputs[k]
            for k in input_names()
        )
        with torch.inference_mode():
            ref = wrapper_fp32(*args32)
            low = wrapper_low(*args_low)
        trt_out = runner(inputs)

        print(f"{name}  patches={inputs['pixel_values'].shape[0]}")
        for i, out_name in enumerate(output_names()):
            r_torch, c_torch = rel_err(ref[i], low[i])
            r_trt, c_trt = rel_err(ref[i], trt_out[i])
            r_pair, c_pair = rel_err(low[i], trt_out[i])
            worst_torch_vs_fp32 = max(worst_torch_vs_fp32, r_torch)
            worst_trt_vs_torch = max(worst_trt_vs_torch, r_pair)
            print(f"  {out_name:<13} torch{precision}/fp32 rel={r_torch:.2e} cos={c_torch:.6f} | "
                  f"TRT/fp32 rel={r_trt:.2e} cos={c_trt:.6f} | "
                  f"TRT/torch{precision} rel={r_pair:.2e} cos={c_pair:.6f}")

    # The real question is whether TensorRT agrees with torch at the SAME
    # precision; drift they share is the cost of the precision, not a TRT bug.
    tol = 3e-2
    verdict = "PASS" if worst_trt_vs_torch < tol else "FAIL"
    print(f"\nTRT vs torch@{precision}:  worst rel {worst_trt_vs_torch:.3e}  -> {verdict} (tol {tol})")
    print(f"torch@{precision} vs fp32: worst rel {worst_torch_vs_fp32:.3e}  "
          f"(cost of {precision} itself, not attributable to TensorRT)\n")
    worst = worst_trt_vs_torch

    print("=== benchmark ===")
    torch_fp16 = wrapper_low
    compiled = torch.compile(torch_fp16, dynamic=True)

    compile_failure_reported = False
    print(f"{'case':<18}{'patches':>8}{'torch':>13}{'compile':>11}{'TRT':>11}{'speedup':>10}")
    for name, image in vision_shapes():
        inputs_h = make_inputs(processor, [image], torch_dtype)
        inputs_f = make_inputs(processor, [image], torch.float32)
        args_h = tuple(inputs_h[k] for k in input_names())
        n = inputs_h["pixel_values"].shape[0]

        with torch.inference_mode():
            t_eager = bench(lambda: torch_fp16(*args_h))
            try:
                t_comp = bench(lambda: compiled(*args_h), iters=20, warmup=4)
            except Exception as exc:
                if not compile_failure_reported:
                    print(f"  (torch.compile unavailable: {type(exc).__name__}: "
                          f"{str(exc).splitlines()[0][:130]})")
                    compile_failure_reported = True
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
           "trt_vs_torch_same_precision_worst_rel": worst_trt_vs_torch,
           "torch_low_vs_fp32_worst_rel": worst_torch_vs_fp32,
           "parity": verdict, "benchmark": rows}
    (BASELINE_DIR / f"phase1_vision_trt_{precision}.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {BASELINE_DIR / f'phase1_vision_trt_{precision}.json'}")

    del model, wrapper_fp32, wrapper_low, tower_low, compiled
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
