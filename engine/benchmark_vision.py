"""Time the TensorRT vision engine against torch at several patch counts.

    python engine/benchmark_vision.py --fp16
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
    bench,
    graph_args,
    make_inputs,
    requested_precision,
)
from qwen3vl import Qwen3VLImageProcessor, load_qwen3vl
from qwen3vl.export_vision import ExportableVisionTower
from tests.fixtures import vision_shapes


def main() -> None:
    precision = requested_precision(sys.argv)
    plan = paths.engine_path(precision)
    if not plan.exists():
        raise SystemExit(f"missing {plan.name} — run engine/build_vision.py --{precision}")

    torch_dtype = TORCH_DTYPES[precision]
    processor = Qwen3VLImageProcessor.from_pretrained(paths.MODEL_DIR)
    runner = TRTVisionRunner(plan)

    model = load_qwen3vl(paths.MODEL_DIR, dtype=torch.float32, device="cuda")
    tower = copy.deepcopy(model.model.visual).to(torch_dtype).eval()
    torch_tower = ExportableVisionTower(tower).eval()
    del model
    torch.cuda.empty_cache()

    compiled = torch.compile(torch_tower, dynamic=True)
    compile_note_shown = False

    print(f"=== benchmark ({precision}) ===")
    print(f"{'case':<18}{'patches':>8}{'torch':>11}{'compile':>11}{'TRT':>11}{'speedup':>10}")

    rows = []
    for name, image in vision_shapes():
        inputs_low = make_inputs(processor, [image], torch_dtype)
        inputs_f32 = make_inputs(processor, [image], torch.float32)
        args = graph_args(inputs_low)
        num_patches = inputs_low["pixel_values"].shape[0]

        with torch.inference_mode():
            t_eager = bench(lambda: torch_tower(*args))
            try:
                t_comp = bench(lambda: compiled(*args), iters=20, warmup=4)
            except Exception as exc:
                if not compile_note_shown:
                    print(f"  (torch.compile unavailable: {type(exc).__name__}: "
                          f"{str(exc).splitlines()[0][:120]})")
                    compile_note_shown = True
                t_comp = float("nan")
        t_trt = bench(lambda: runner(inputs_f32))

        best_torch = min(t_eager, t_comp) if t_comp == t_comp else t_eager
        speedup = best_torch / t_trt
        rows.append({
            "case": name, "patches": num_patches,
            "torch_ms": round(t_eager, 3), "compile_ms": round(t_comp, 3),
            "trt_ms": round(t_trt, 3), "speedup_vs_best_torch": round(speedup, 2),
        })
        print(f"{name:<18}{num_patches:>8}{t_eager:>10.2f}m{t_comp:>10.2f}m"
              f"{t_trt:>10.2f}m{speedup:>9.2f}x")

    paths.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.BASELINE_DIR / f"vision_benchmark_{precision}.json"
    out.write_text(json.dumps({"precision": precision, "benchmark": rows}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
