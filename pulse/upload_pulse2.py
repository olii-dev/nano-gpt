#!/usr/bin/env python3
"""Upload Lattice Pulse 2 LoRA adapter to Hugging Face Hub.

Pulse 2 was trained as QLoRA — this uploads the ~170MB adapter only.
Loaders still need the base model (Qwen/Qwen3-8B) unless you merge first.

Usage:
  hf auth login
  python -m pulse.upload_pulse2
  python -m pulse.upload_pulse2 --merge   # merge + upload full ~16GB weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pulse.config_pulse2 import BASE_MODEL, HF_REPO_ID

DEFAULT_ADAPTER = Path(
    "/Users/olimebberson/Downloads/Firefox Downloads/results/lattice-pulse-2-8b-lora/checkpoint-400"
)


def upload_adapter(adapter: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi, create_repo

    if not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter}")

    create_repo(repo_id, exist_ok=True, private=False)
    api = HfApi()
    print(f"Uploading adapter {adapter} → {repo_id} ...")
    api.upload_folder(
        folder_path=str(adapter),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["optimizer.pt", "rng_state.pth", "scaler.pt", "scheduler.pt", "training_args.bin", ".DS_Store"],
    )
    readme = f"""---
library_name: peft
base_model: {BASE_MODEL}
license: apache-2.0
tags:
  - lora
  - qwen3
  - lattice
---

# Lattice Pulse 2 (8B) — LoRA adapter

QLoRA fine-tune of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) for Lattice Systems.

This repo is the **adapter only** (~170MB). Inference still loads the base model, then applies this LoRA:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")
tok = AutoTokenizer.from_pretrained("{repo_id}")
```

Or locally: `python -m pulse.chat_pulse2 --adapter /path/to/checkpoint-400`
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done: https://huggingface.co/{repo_id}")


def merge_and_upload(adapter: Path, repo_id: str, out_dir: Path) -> None:
    """Merge LoRA into fp16 base and upload full weights (Pulse-1 style, ~16GB)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi, create_repo

    print(f"Loading base {BASE_MODEL} (fp16, CPU) — needs ~16GB RAM...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"Applying LoRA from {adapter} ...")
    model = PeftModel.from_pretrained(base, str(adapter))
    print("Merging...")
    model = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model → {out_dir}")
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))

    merged_repo = repo_id if repo_id.endswith("-merged") else f"{repo_id}-merged"
    create_repo(merged_repo, exist_ok=True, private=False)
    api = HfApi()
    print(f"Uploading merged weights → {merged_repo} (large)...")
    api.upload_folder(folder_path=str(out_dir), repo_id=merged_repo, repo_type="model")
    print(f"Done: https://huggingface.co/{merged_repo}")


def main() -> None:
    p = argparse.ArgumentParser(description="Upload Pulse 2 to Hugging Face")
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--repo", default=HF_REPO_ID)
    p.add_argument("--merge", action="store_true", help="Merge LoRA into full weights then upload")
    p.add_argument(
        "--merge-out",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "lattice-pulse-2-8b-merged",
    )
    args = p.parse_args()

    if args.merge:
        merge_and_upload(args.adapter, args.repo, args.merge_out)
    else:
        upload_adapter(args.adapter, args.repo)


if __name__ == "__main__":
    main()
