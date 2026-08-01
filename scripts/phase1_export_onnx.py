"""Phase 1a: validate the exportable vision tower, then export it to ONNX.

Two gates before TensorRT is allowed to enter the picture:
  1. `ExportableVisionTower` must reproduce the eager tower (same weights, same
     math, different packaging).
  2. The exported ONNX must reproduce the wrapper under onnxruntime, at several
     patch counts, to prove the dynamic axis actually works.
"""

from __future__ import annotations

import copy
import gc
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _fixtures import BASELINE_DIR, MODEL_DIR, vision_shapes

from qwen3vl import Qwen3VLImageProcessor, load_qwen3vl
from qwen3vl.export_vision import (
    ExportableVisionTower,
    build_vision_inputs,
    dynamic_axes,
    input_names,
    output_names,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# TensorRT 11 builds strongly-typed networks only: there is no FP16/BF16 builder
# flag any more, so the graph itself must carry the precision we want to run.
PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def onnx_path(precision: str) -> Path:
    return ARTIFACT_DIR / f"vision_tower_{precision}.onnx"


def requested_precisions() -> list[str]:
    picked = [p for p in PRECISIONS if f"--{p}" in sys.argv]
    return picked or ["fp32", "fp16"]


def make_case(
    processor: Qwen3VLImageProcessor, images, device, dtype: torch.dtype = DTYPE
) -> dict[str, torch.Tensor]:
    """Preprocess images and build every tensor the exported graph consumes."""
    vision = processor(images)
    cfg_inputs = build_vision_inputs(
        vision["image_grid_thw"],
        num_grid_per_side=48,
        spatial_merge_size=2,
        device=device,
    )
    cfg_inputs["pos_weights"] = cfg_inputs["pos_weights"].to(dtype)
    return {
        "pixel_values": vision["pixel_values"].to(device, dtype),
        "grid_thw": vision["image_grid_thw"].to(device),
        **cfg_inputs,
    }


def graph_args(case: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(case[name] for name in input_names())


def report(label: str, a: torch.Tensor, b: torch.Tensor) -> float:
    """Print absolute and scale-relative error; return the relative one.

    Activation magnitudes vary a lot across these tensors, so absolute error
    alone is not comparable between outputs — everything is gated on
    max_abs / max|reference|.
    """
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    scale = a.abs().max().clamp(min=1e-6)
    rel = float(diff.max() / scale)
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    print(
        f"  {label:<16} max_abs={diff.max():.3e}  rel={rel:.3e}  "
        f"mean={diff.mean():.3e}  cos={cos:.8f}  |ref|max={scale:.1f}"
    )
    return rel


def gate_wrapper_parity(model, processor) -> None:
    print("=== gate 1: wrapper vs eager tower ===")
    wrapper = ExportableVisionTower(model.model.visual).eval()
    worst = 0.0

    for name, image in vision_shapes():
        case = make_case(processor, [image], DEVICE)
        with torch.inference_mode():
            eager_merged, eager_deep = model.model.visual(case["pixel_values"], case["grid_thw"])
            out = wrapper(*graph_args(case))

        print(f"{name}  patches={case['pixel_values'].shape[0]}")
        worst = max(worst, report("merged", eager_merged, out[0]))
        for i in range(3):
            worst = max(worst, report(f"deepstack_{i}", eager_deep[i], out[1 + i]))

    # Multi-image is the case the segment mask exists for.
    imgs = [img for _, img in vision_shapes()[:3]]
    case = make_case(processor, imgs, DEVICE)
    with torch.inference_mode():
        eager_merged, eager_deep = model.model.visual(case["pixel_values"], case["grid_thw"])
        out = wrapper(*graph_args(case))
    print(f"multi_image(3)  patches={case['pixel_values'].shape[0]}")
    worst = max(worst, report("merged", eager_merged, out[0]))
    for i in range(3):
        worst = max(worst, report(f"deepstack_{i}", eager_deep[i], out[1 + i]))

    tol = 1e-3
    print(f"gate 1 {'PASS' if worst < tol else 'FAIL'}  (worst {worst:.3e}, tol {tol})\n")
    if worst >= tol:
        raise SystemExit("wrapper parity failed — fix before exporting")


def export_onnx(model, processor, precision: str) -> None:
    out_path = onnx_path(precision)
    if out_path.exists() and "--force-export" not in sys.argv:
        print(f"=== reusing {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB) ===")
        print("    pass --force-export to re-export\n")
        return

    dtype = PRECISIONS[precision]
    print(f"=== exporting ONNX ({precision}) ===")
    ARTIFACT_DIR.mkdir(exist_ok=True)

    # Export on CPU: avoids CUDA-specific SDPA kernels leaking into the graph.
    # A fresh module per precision keeps the loaded fp32 weights untouched.
    tower = copy.deepcopy(model.model.visual).to("cpu").to(dtype).eval()
    wrapper = ExportableVisionTower(tower).eval()
    case = make_case(processor, [vision_shapes()[1][1]], "cpu", dtype)

    torch.onnx.export(
        wrapper,
        graph_args(case),
        str(out_path),
        input_names=input_names(),
        output_names=output_names(),
        dynamic_axes=dynamic_axes(),
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    import onnx

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    ops: dict[str, int] = {}
    for node in onnx_model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    top = sorted(ops.items(), key=lambda kv: -kv[1])[:10]
    print(f"checker OK. {len(onnx_model.graph.node)} nodes, top ops: {top}\n")

    del tower, wrapper
    gc.collect()


def gate_onnxruntime(model, processor) -> None:
    print("=== gate 2: onnxruntime vs wrapper (dynamic shapes) ===")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path("fp32")), providers=["CPUExecutionProvider"])
    # Compare CPU-to-CPU: mixing devices here would measure GPU-vs-CPU float
    # accumulation instead of whether the exported graph is faithful.
    wrapper = ExportableVisionTower(model.model.visual).eval().to("cpu")
    worst = 0.0

    for name, image in vision_shapes():
        case = make_case(processor, [image], "cpu")
        feeds = {key: case[key].numpy() for key in input_names()}

        ort_out = session.run(output_names(), feeds)
        with torch.inference_mode():
            torch_out = wrapper(*graph_args(case))

        print(f"{name}  patches={case['pixel_values'].shape[0]}")
        for i, out_name in enumerate(output_names()):
            worst = max(
                worst,
                report(out_name, torch_out[i], torch.from_numpy(np.asarray(ort_out[i]))),
            )

    wrapper.to(DEVICE)
    tol = 2e-3
    print(f"gate 2 {'PASS' if worst < tol else 'FAIL'}  (worst rel {worst:.3e}, tol {tol})\n")
    if worst >= tol:
        raise SystemExit("onnxruntime parity failed")


def main() -> None:
    precisions = requested_precisions()
    print(f"device={DEVICE} reference dtype={DTYPE} exporting={precisions}\n")
    model = load_qwen3vl(MODEL_DIR, dtype=DTYPE, device=DEVICE)
    processor = Qwen3VLImageProcessor.from_pretrained(MODEL_DIR)

    gate_wrapper_parity(model, processor)
    for precision in precisions:
        export_onnx(model, processor, precision)
    if "fp32" in precisions:
        gate_onnxruntime(model, processor)

    print("Phase 1a complete. ONNX ready for TensorRT engine build.")


if __name__ == "__main__":
    main()
