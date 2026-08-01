"""Phase 1c: end-to-end check — does the TensorRT vision engine change answers?

Feature-level relative error is a proxy. For a VLM the gate that matters is
whether the generated text changes.

Comparing TRT against the bf16 torch pipeline alone would be misleading: the
bf16 pipeline is itself an approximation of the fp32 model. So all pipelines
are scored against a **fp32 torch reference**:

    A  torch fp32                      ground truth
    B  torch bf16                      the shipping baseline
    C  torch bf16 decoder + TRT vision  what we built

If B already diverges from A, token-identity is not a reachable bar for any
reduced-precision engine, and the honest question becomes which of B and C
tracks A more closely.

The vision swap in C is also exactly the handoff Phase 2 needs: the engine
produces merged tokens plus three DeepStack tensors, the decoder consumes them.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths
from engine.runtime import TRTVisionRunner, requested_precision
from tests.fixtures import BASELINE_DIR, MODEL_DIR, prompt_cases

from qwen3vl import GenerationConfig, Qwen3VLProcessor, generate, load_qwen3vl
from qwen3vl.export_vision import build_vision_inputs, input_names

MAX_NEW_TOKENS = 48


def make_trt_image_features(runner: TRTVisionRunner, out_dtype: torch.dtype):
    """Drop-in replacement for `Qwen3VLModel.get_image_features` backed by TensorRT."""

    def get_image_features(pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        extra = build_vision_inputs(image_grid_thw, 48, 2, device="cuda")
        inputs = {"pixel_values": pixel_values.float(), **extra}
        merged, d0, d1, d2 = runner({k: inputs[k] for k in input_names()})
        return merged.to(out_dtype), [d0.to(out_dtype), d1.to(out_dtype), d2.to(out_dtype)]

    return get_image_features


def run_cases(model, processor, label: str) -> dict[str, tuple[list[int], str, float]]:
    config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    results = {}
    for name, messages in prompt_cases():
        batch = processor.apply_chat_template(messages, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        out = generate(model, **batch, config=config)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        tokens = out.generated[0].cpu().tolist()
        results[name] = (tokens, processor.decode(out.generated[0]), elapsed)
        print(f"  [{label:<9}] {name:<14} {len(tokens):>3} tok in {elapsed:>5.2f}s")
    return results


def agreement(reference: dict, candidate: dict) -> tuple[int, int, list[str]]:
    """Count cases matching the reference; return (matches, total, notes)."""
    matches, notes = 0, []
    for name, _ in prompt_cases():
        ref_tok = reference[name][0]
        cand_tok = candidate[name][0]
        common = min(len(ref_tok), len(cand_tok))
        first_diff = next((i for i in range(common) if ref_tok[i] != cand_tok[i]), None)
        if first_diff is None and len(ref_tok) == len(cand_tok):
            matches += 1
            notes.append(f"{name}: match")
        else:
            notes.append(f"{name}: diverges at token {first_diff}")
    return matches, len(prompt_cases()), notes


def main() -> None:
    precision = requested_precision(sys.argv)
    plan = paths.engine_path(precision)
    if not plan.exists():
        raise SystemExit(f"missing {plan.name} — run phase1_build_engine.py --{precision}")

    processor = Qwen3VLProcessor.from_pretrained(MODEL_DIR)

    print("=== A: torch fp32 (ground truth) ===")
    model = load_qwen3vl(MODEL_DIR, dtype=torch.float32, device="cuda")
    results_fp32 = run_cases(model, processor, "fp32")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== B: torch bf16 (shipping baseline) ===")
    model = load_qwen3vl(MODEL_DIR, dtype=torch.bfloat16, device="cuda")
    results_bf16 = run_cases(model, processor, "bf16")

    print(f"\n=== C: torch bf16 decoder + TRT {precision} vision ===")
    runner = TRTVisionRunner(plan)
    original = model.model.get_image_features
    model.model.get_image_features = make_trt_image_features(runner, torch.bfloat16)
    results_trt = run_cases(model, processor, f"TRT-{precision}")
    model.model.get_image_features = original

    print("\n=== agreement with fp32 ground truth ===")
    b_match, total, b_notes = agreement(results_fp32, results_bf16)
    c_match, _, c_notes = agreement(results_fp32, results_trt)

    print(f"\nB  torch bf16          {b_match}/{total} cases match fp32")
    for note in b_notes:
        print(f"     {note}")
    print(f"\nC  bf16 + TRT {precision:<5}   {c_match}/{total} cases match fp32")
    for note in c_notes:
        print(f"     {note}")

    print("\n=== generated text ===")
    for name, _ in prompt_cases():
        print(f"\n[{name}]")
        for label, res in (("fp32", results_fp32), ("bf16", results_bf16), (f"TRT-{precision}", results_trt)):
            print(f"  {label:<10} {res[name][1].strip()[:140]!r}")

    lines = [
        "# Phase 1c — end-to-end vision engine check",
        "",
        f"TensorRT vision precision: {precision}. Decoder stays torch bf16 in B and C.",
        "",
        "| pipeline | cases matching fp32 |",
        "|---|---|",
        f"| B: torch bf16 (shipping baseline) | {b_match}/{total} |",
        f"| C: torch bf16 + TRT {precision} vision | {c_match}/{total} |",
        "",
        "Scoring against fp32 rather than against the bf16 pipeline matters: bf16 is",
        "itself an approximation, so any divergence it already shows is the floor for",
        "what a reduced-precision engine can be held to.",
        "",
    ]
    for name, _ in prompt_cases():
        lines.append(f"### {name}")
        for label, res in (("fp32", results_fp32), ("bf16", results_bf16), (f"TRT-{precision}", results_trt)):
            lines.append(f"- **{label}**: `{res[name][1].strip()[:200]}`")
        lines.append("")
    (BASELINE_DIR / f"phase1_end2end_{precision}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {BASELINE_DIR / f'phase1_end2end_{precision}.md'}")


if __name__ == "__main__":
    main()
