"""Download Qwen3-VL-2B-Instruct weights into models/ as a plain snapshot."""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEST = Path(__file__).resolve().parent.parent / "models" / "Qwen3-VL-2B-Instruct"

# Skip the duplicate .pth/.msgpack/.h5 copies of the same tensors.
IGNORE = ["*.pth", "*.msgpack", "*.h5", "*.ot"]


def main() -> None:
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=DEST,
        ignore_patterns=IGNORE,
        max_workers=8,
    )
    print(f"snapshot at: {path}")
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"total size: {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
