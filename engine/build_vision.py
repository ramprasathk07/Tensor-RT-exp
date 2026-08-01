"""Build the TensorRT vision engine from an exported ONNX graph.

Building only — validation lives in `tests/test_engine_parity.py` and timing in
`engine/benchmark_vision.py`.

    python engine/build_vision.py            # fp16 (default)
    python engine/build_vision.py --bf16
    python engine/build_vision.py --rebuild
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths
from engine.runtime import TRT_LOGGER, requested_precision

# Optimization profile: covers 256x256 up to ~1280x720 images with headroom.
MIN_PATCHES, OPT_PATCHES, MAX_PATCHES = 256, 1024, 6144

# Shape of each graph input as a function of the patch count.
INPUT_SHAPES = {
    "pixel_values": lambda n: (n, 1536),
    "pos_indices": lambda n: (4, n),
    "pos_weights": lambda n: (4, n),
    "vision_position_ids": lambda n: (n, 2),
    "segment_ids": lambda n: (n,),
}


def build_engine(precision: str, rebuild: bool = False) -> Path:
    out_path = paths.engine_path(precision)
    if out_path.exists() and not rebuild:
        print(f"=== reusing {out_path.name} ({out_path.stat().st_size / 1e6:.0f} MB) ===")
        print("    pass --rebuild to force")
        return out_path

    onnx_file = paths.onnx_path(precision)
    if not onnx_file.exists():
        raise SystemExit(f"missing {onnx_file.name} — run export/vision_onnx.py --{precision}")

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
    for name, shape_fn in INPUT_SHAPES.items():
        profile.set_shape(name, shape_fn(MIN_PATCHES), shape_fn(OPT_PATCHES), shape_fn(MAX_PATCHES))
    config.add_optimization_profile(profile)

    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")

    paths.ARTIFACT_DIR.mkdir(exist_ok=True)
    out_path.write_bytes(serialized)
    print(f"    built in {(time.time() - started) / 60:.1f} min -> {out_path.name} "
          f"({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


def main() -> None:
    precision = requested_precision(sys.argv)
    plan = build_engine(precision, rebuild="--rebuild" in sys.argv)
    print(f"\nengine ready: {plan}")
    print(f"next: python tests/test_engine_parity.py --{precision}")


if __name__ == "__main__":
    main()
