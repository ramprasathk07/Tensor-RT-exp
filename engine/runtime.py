"""TensorRT execution helpers shared by the build, benchmark, profile and test entry points."""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch

import paths  # noqa: F401  - repo-root import bootstrap
from qwen3vl.export_vision import build_vision_inputs, input_names, output_names

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

TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def requested_precision(argv: list[str], default: str = "fp16") -> str:
    for precision in TORCH_DTYPES:
        if f"--{precision}" in argv:
            return precision
    return default


class TRTVisionRunner:
    """Thin execution wrapper around the vision engine: torch tensors in and out."""

    def __init__(self, plan_path: Path) -> None:
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
        self.context = self.engine.create_execution_context()

    def __call__(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        for name in input_names():
            if not self.context.set_input_shape(name, tuple(inputs[name].shape)):
                raise RuntimeError(
                    f"shape rejected for {name}: {tuple(inputs[name].shape)} "
                    "(outside the engine's optimization profile?)"
                )

        # Cast to whatever the engine declares, so callers can pass fp32 and let
        # an fp16 engine consume it without a separate preprocessing path.
        held = []
        for name in input_names():
            tensor = inputs[name].to(TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]).contiguous()
            held.append(tensor)
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
        return outputs


def make_inputs(processor, images, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Preprocess images into the full set of graph inputs, on CUDA."""
    vision = processor(images)
    extra = build_vision_inputs(vision["image_grid_thw"], 48, 2, device="cuda")
    return {
        "pixel_values": vision["pixel_values"].to("cuda", dtype),
        "grid_thw": vision["image_grid_thw"].to("cuda"),
        **extra,
    }


def graph_args(case: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(case[name] for name in input_names())


def bench(fn, iters: int = 30, warmup: int = 8) -> float:
    """Median milliseconds per call, measured with CUDA events."""
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


def rel_error(reference: torch.Tensor, other: torch.Tensor) -> tuple[float, float]:
    """Scale-relative max error and cosine similarity between two tensors."""
    a, b = reference.float(), other.float()
    rel = float((a - b).abs().max() / a.abs().max().clamp(min=1e-6))
    cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    return rel, cos
