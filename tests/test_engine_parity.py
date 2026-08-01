"""Validate the TensorRT vision engine numerically against torch.

Three-way comparison. Checking TensorRT against fp32 alone cannot separate
"TensorRT is wrong" from "reduced precision costs this much", so torch at the
*same* precision is measured as the control:

    fp32 reference   <->   torch @ precision   <->   TensorRT @ precision

Drift that torch shows too is the price of the precision. Only the
TensorRT-vs-torch column is evidence about TensorRT itself.

    python tests/test_engine_parity.py --fp16
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths
from engine.runtime import (
    TORCH_DTYPES,
    TRTVisionRunner,
    graph_args,
    make_inputs,
    rel_error,
    requested_precision,
)
from qwen3vl import Qwen3VLImageProcessor, load_qwen3vl
from qwen3vl.export_vision import ExportableVisionTower, output_names
from tests.fixtures import vision_shapes

TOLERANCE = 3e-2


def main() -> None:
    precision = requested_precision(sys.argv)
    plan = paths.engine_path(precision)
    if not plan.exists():
        raise SystemExit(f"missing {plan.name} — run engine/build_vision.py --{precision}")

    torch_dtype = TORCH_DTYPES[precision]
    processor = Qwen3VLImageProcessor.from_pretrained(paths.MODEL_DIR)
    runner = TRTVisionRunner(plan)

    print(f"=== fp32 reference vs torch {precision} vs TRT {precision} ===")
    model = load_qwen3vl(paths.MODEL_DIR, dtype=torch.float32, device="cuda")
    wrapper_fp32 = ExportableVisionTower(model.model.visual).eval()
    tower_low = copy.deepcopy(model.model.visual).to(torch_dtype).eval()
    wrapper_low = ExportableVisionTower(tower_low).eval()

    worst_trt_vs_torch = 0.0
    worst_torch_vs_fp32 = 0.0

    for name, image in vision_shapes():
        inputs = make_inputs(processor, [image], torch.float32)
        args_low = tuple(
            t.to(torch_dtype) if t.is_floating_point() else t for t in graph_args(inputs)
        )
        with torch.inference_mode():
            ref = wrapper_fp32(*graph_args(inputs))
            low = wrapper_low(*args_low)
        trt_out = runner(inputs)

        print(f"{name}  patches={inputs['pixel_values'].shape[0]}")
        for i, out_name in enumerate(output_names()):
            r_torch, c_torch = rel_error(ref[i], low[i])
            r_trt, c_trt = rel_error(ref[i], trt_out[i])
            r_pair, c_pair = rel_error(low[i], trt_out[i])
            worst_torch_vs_fp32 = max(worst_torch_vs_fp32, r_torch)
            worst_trt_vs_torch = max(worst_trt_vs_torch, r_pair)
            print(f"  {out_name:<13} torch/fp32 rel={r_torch:.2e} cos={c_torch:.6f} | "
                  f"TRT/fp32 rel={r_trt:.2e} cos={c_trt:.6f} | "
                  f"TRT/torch rel={r_pair:.2e} cos={c_pair:.6f}")

    verdict = "PASS" if worst_trt_vs_torch < TOLERANCE else "FAIL"
    print(f"\nTRT vs torch@{precision}:  worst rel {worst_trt_vs_torch:.3e}  "
          f"-> {verdict} (tol {TOLERANCE})")
    print(f"torch@{precision} vs fp32: worst rel {worst_torch_vs_fp32:.3e}  "
          f"(cost of {precision} itself, not attributable to TensorRT)")
    print("\nFeature-level error is a max-over-outliers metric; the decisive check "
          "is tests/test_end2end.py, which compares generated tokens.")

    paths.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    (paths.BASELINE_DIR / f"engine_parity_{precision}.json").write_text(
        json.dumps(
            {
                "precision": precision,
                "trt_vs_torch_same_precision_worst_rel": worst_trt_vs_torch,
                "torch_low_vs_fp32_worst_rel": worst_torch_vs_fp32,
                "verdict": verdict,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
