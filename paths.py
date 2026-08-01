"""Canonical project paths, and repo-root import bootstrap.

Entry points live one directory below the repo root, so running
`python tests/smoke.py` puts `tests/` on `sys.path` rather than the root.
Each entry point therefore opens with::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths

after which `from qwen3vl import ...`, `from engine.runtime import ...` and
`from tests.fixtures import ...` all resolve regardless of the launch
directory. Modules that are only ever imported (never run directly) can just
`import paths`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_DIR = REPO_ROOT / "models" / "Qwen3-VL-2B-Instruct"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
BASELINE_DIR = REPO_ROOT / "docs" / "baselines"
TENSOR_DIR = BASELINE_DIR / "tensors"


def onnx_path(precision: str) -> Path:
    return ARTIFACT_DIR / f"vision_tower_{precision}.onnx"


def engine_path(precision: str) -> Path:
    return ARTIFACT_DIR / f"vision_tower_{precision}.plan"
