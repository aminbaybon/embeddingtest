"""Download BGE-M3 and EmbeddingGemma-2 into ./models."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "embeddinggemma-2": "google/embeddinggemma-2",
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Download embedding checkpoints into ./models")
    parser.add_argument("names", nargs="*", default=list(MODELS), help="bge-m3 and/or embeddinggemma-2")
    args = parser.parse_args()
    for name in args.names:
        repo = MODELS.get(name, name)
        dest = ROOT / "models" / (name if name in MODELS else repo.split("/")[-1])
        dest.mkdir(parents=True, exist_ok=True)
        print(f"downloading {repo} -> {dest}", flush=True)
        path = snapshot_download(repo_id=repo, local_dir=str(dest), max_workers=2, etag_timeout=60.0)
        print(f"ready {path}", flush=True)


if __name__ == "__main__":
    main()
