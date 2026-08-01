"""Gate 0: freeze the reference oracle.

Dumps fp32 reference tensors for the fixed prompt fixtures plus an environment
record. Every TensorRT / TensorRT-LLM engine built later is validated against
these files, so no engine check ever needs `transformers` or the torch model
loaded again.

Outputs into `docs/baselines/`:
    environment.txt          versions, GPU, driver
    tensors/<case>.pt        inputs, vision features, positions, logits
    freeze_report.md         human-readable summary of what was captured
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures import BASELINE_DIR, MODEL_DIR, prompt_cases

from qwen3vl import Qwen3VLProcessor, load_qwen3vl

DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def write_environment(path: Path) -> None:
    import transformers

    lines = [f"captured_utc      {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    lines.append(f"platform          {platform.platform()}")
    lines.append(f"python            {sys.version.split()[0]}")
    lines.append(f"torch             {torch.__version__}")
    lines.append(f"torch_cuda        {torch.version.cuda}")
    lines.append(f"transformers      {transformers.__version__}")
    if torch.cuda.is_available():
        lines.append(f"gpu               {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        lines.append(f"compute_cap       {cap[0]}.{cap[1]}")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        lines.append(f"vram_gb           {total:.1f}")
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        lines.append(f"nvidia_driver     {driver}")
    except Exception:
        lines.append("nvidia_driver     unknown")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print("\n".join(f"  {line}" for line in lines))


@torch.inference_mode()
def freeze_case(model, processor, name: str, messages: list[dict], out_dir: Path) -> dict:
    batch = processor.apply_chat_template(messages, device=DEVICE)

    record: dict[str, torch.Tensor | list] = {
        "input_ids": batch["input_ids"].cpu(),
        "attention_mask": batch["attention_mask"].cpu(),
    }
    has_image = "pixel_values" in batch
    if has_image:
        record["pixel_values"] = batch["pixel_values"].cpu()
        record["image_grid_thw"] = batch["image_grid_thw"].cpu()

        merged, deepstack = model.model.get_image_features(
            batch["pixel_values"], batch["image_grid_thw"]
        )
        record["vision_merged"] = merged.cpu()
        for i, d in enumerate(deepstack):
            record[f"vision_deepstack_{i}"] = d.cpu()

    # 3-D M-RoPE positions: the single most error-prone thing to reproduce in
    # another runtime, so freeze them explicitly rather than recomputing later.
    positions = model.model.compute_3d_position_ids(
        input_ids=batch["input_ids"],
        inputs_embeds=model.model.language_model.embed_tokens(batch["input_ids"]),
        image_grid_thw=batch.get("image_grid_thw"),
        attention_mask=batch["attention_mask"],
        past_key_values=None,
    )
    record["position_ids_3d"] = positions.cpu()

    logits = model(**batch)
    record["logits"] = logits.float().cpu()
    record["argmax_last"] = logits[:, -1, :].argmax(-1).cpu()

    out_path = out_dir / f"{name}.pt"
    torch.save(record, out_path)

    summary = {
        "case": name,
        "prompt_tokens": int(batch["input_ids"].shape[1]),
        "patches": int(batch["pixel_values"].shape[0]) if has_image else 0,
        "grid": batch["image_grid_thw"].tolist() if has_image else [],
        "vision_tokens": int(record["vision_merged"].shape[0]) if has_image else 0,
        "max_position": int(positions.max()),
        "logits_shape": list(logits.shape),
        "last_argmax": int(record["argmax_last"][0]),
        "file_mb": round(out_path.stat().st_size / 1e6, 1),
    }
    print(f"  {name:14} tokens={summary['prompt_tokens']:<5} patches={summary['patches']:<6} "
          f"max_pos={summary['max_position']:<5} argmax={summary['last_argmax']:<7} "
          f"({summary['file_mb']} MB)")
    return summary


def main() -> None:
    tensor_dir = BASELINE_DIR / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    print("=== environment ===")
    write_environment(BASELINE_DIR / "environment.txt")

    print(f"\n=== loading reference model (fp32, {DEVICE}) ===")
    model = load_qwen3vl(MODEL_DIR, dtype=DTYPE, device=DEVICE)
    processor = Qwen3VLProcessor.from_pretrained(MODEL_DIR)

    print("\n=== freezing cases ===")
    summaries = [
        freeze_case(model, processor, name, messages, tensor_dir)
        for name, messages in prompt_cases()
    ]

    report = ["# Gate 0 — frozen reference", "", f"dtype: {DTYPE}, device: {DEVICE}", "",
              "| case | prompt tokens | patches | grid | vision tokens | max pos | last argmax |",
              "|---|---|---|---|---|---|---|"]
    for s in summaries:
        report.append(
            f"| `{s['case']}` | {s['prompt_tokens']} | {s['patches']} | "
            f"{s['grid']} | {s['vision_tokens']} | {s['max_position']} | {s['last_argmax']} |"
        )
    report += [
        "",
        "Tensors in `tensors/<case>.pt` (gitignored). Each holds inputs,",
        "`vision_merged`, `vision_deepstack_{0,1,2}`, `position_ids_3d`, and `logits`.",
        "",
        "Note the `max pos` column: it is well below the prompt token count for image",
        "cases because an image advances the M-RoPE counter by `max(h,w)//2`, not by",
        "its token count. Any runtime that assumes `position == seq_len` will diverge.",
    ]
    (BASELINE_DIR / "freeze_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    (BASELINE_DIR / "freeze_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {BASELINE_DIR / 'freeze_report.md'}")


if __name__ == "__main__":
    main()
