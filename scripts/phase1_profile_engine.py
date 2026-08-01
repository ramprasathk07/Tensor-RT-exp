"""Phase 1d: per-layer profile of the TensorRT vision engine.

Answers the question the kernel roadmap depends on: where does the engine's
time actually go, and which op sequences did TensorRT decline to fuse? Layers
that survive as separate, expensive kernels are the candidates worth writing by
hand; everything TensorRT already fused is not.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _fixtures import BASELINE_DIR, MODEL_DIR, vision_shapes
from phase1_build_engine import TRTVisionRunner, engine_path, make_inputs, requested_precision

from qwen3vl import Qwen3VLImageProcessor
from qwen3vl.export_vision import input_names


class LayerProfiler(trt.IProfiler):
    """Accumulates per-layer milliseconds across runs."""

    def __init__(self) -> None:
        super().__init__()
        self.totals: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.totals[layer_name] += ms
        self.calls[layer_name] += 1


def classify(layer_name: str) -> str:
    """Bucket a TensorRT layer name into a rough op family."""
    name = layer_name.lower()
    if "myelin" in name:
        return "myelin (fused subgraph)"
    if any(k in name for k in ("gemm", "matmul", "fc", "conv", "convolution")):
        return "gemm / conv"
    if any(k in name for k in ("softmax", "attention", "bmm")):
        return "attention"
    if any(k in name for k in ("layernorm", "norm", "reduce")):
        return "normalization"
    if any(k in name for k in ("gather", "slice", "concat", "shuffle", "transpose", "reformat")):
        return "layout / gather"
    if any(k in name for k in ("elementwise", "add", "mul", "activation", "gelu", "cast")):
        return "elementwise"
    return "other"


def main() -> None:
    precision = requested_precision()
    plan = engine_path(precision)
    if not plan.exists():
        raise SystemExit(f"missing {plan.name} — run phase1_build_engine.py --{precision}")

    processor = Qwen3VLImageProcessor.from_pretrained(MODEL_DIR)
    runner = TRTVisionRunner(plan)

    # Profile at the optimization-profile's opt point, where tactics are tuned.
    name, image = vision_shapes()[1]
    inputs = make_inputs(processor, [image], torch.float32)
    num_patches = inputs["pixel_values"].shape[0]
    print(f"=== profiling {plan.name} at {name} ({num_patches} patches) ===\n")

    for _ in range(5):
        runner({k: inputs[k] for k in input_names()})

    # Keep a reference for the context's lifetime; TensorRT rejects a null
    # profiler, so detaching is done by dropping the context instead.
    profiler = LayerProfiler()
    runner.context.profiler = profiler
    runs = 20
    for _ in range(runs):
        runner({k: inputs[k] for k in input_names()})

    total_ms = sum(profiler.totals.values()) / runs
    print(f"total engine time: {total_ms:.2f} ms/run across {len(profiler.totals)} profiled layers\n")

    by_family: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for layer, ms in profiler.totals.items():
        by_family[classify(layer)] += ms / runs
        counts[classify(layer)] += 1

    print(f"{'op family':<26}{'ms/run':>10}{'% total':>10}{'layers':>9}")
    for family, ms in sorted(by_family.items(), key=lambda kv: -kv[1]):
        print(f"{family:<26}{ms:>10.3f}{100 * ms / total_ms:>9.1f}%{counts[family]:>9}")

    print(f"\n{'top 20 individual layers':<64}{'ms/run':>9}{'%':>7}")
    top = sorted(profiler.totals.items(), key=lambda kv: -kv[1])[:20]
    for layer, ms in top:
        per_run = ms / runs
        short = re.sub(r"\s+", " ", layer)[:62]
        print(f"{short:<64}{per_run:>9.3f}{100 * per_run / total_ms:>6.1f}%")

    lines = [
        f"# Phase 1d — TensorRT vision engine layer profile ({precision})",
        "",
        f"Engine `{plan.name}` at {name} ({num_patches} patches), "
        f"{runs} runs, {total_ms:.2f} ms/run total.",
        "",
        "| op family | ms/run | % total | layers |",
        "|---|---|---|---|",
    ]
    for family, ms in sorted(by_family.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {family} | {ms:.3f} | {100 * ms / total_ms:.1f}% | {counts[family]} |")
    lines += ["", "## Top layers", "", "| layer | ms/run | % |", "|---|---|---|"]
    for layer, ms in top:
        lines.append(f"| `{re.sub(r'\\s+', ' ', layer)[:80]}` | {ms / runs:.3f} | "
                     f"{100 * (ms / runs) / total_ms:.1f}% |")

    out = BASELINE_DIR / f"phase1_layer_profile_{precision}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
