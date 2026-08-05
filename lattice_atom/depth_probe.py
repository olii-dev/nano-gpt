"""Print param count for nanochat models across depths.

nanochat has no published sizing table — we measure. Formula (verified
against scripts/base_train.py:build_model_meta):
  n_layer   = depth
  base_dim  = depth * aspect_ratio (default 64)
  n_embd    = ceil(base_dim / head_dim) * head_dim   (head_dim default 128)
  n_head    = n_embd / head_dim
  vocab     = 32768

Usage (ON THE AZURE VM — instantiates models, do NOT run on Mac):
  cd nanochat && uv run python ../lattice_atom/depth_probe.py
"""
import sys
import math
from pathlib import Path

# Run from inside the nanochat/ dir so `from nanochat.gpt import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nanochat"))

from nanochat.gpt import GPT, GPTConfig

VOCAB = 32768
HEAD_DIM = 128
ASPECT_RATIO = 64


def config_for_depth(depth: int, aspect: int = ASPECT_RATIO,
                     head_dim: int = HEAD_DIM, seq_len: int = 2048) -> GPTConfig:
    """Mirror scripts/base_train.py:build_model_meta exactly."""
    base_dim = depth * aspect
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim
    return GPTConfig(
        sequence_len=seq_len,
        vocab_size=VOCAB,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern="SSSL",
    )


def main() -> None:
    print(f"{'depth':>5} {'n_embd':>6} {'n_head':>6} {'params':>14} {'scaling':>12}")
    print("-" * 50)
    for depth in [16, 18, 20, 22, 24, 26, 28, 30, 32]:
        cfg = config_for_depth(depth)
        model = GPT(cfg)
        total = sum(p.numel() for p in model.parameters())
        scaling = model.num_scaling_params()
        marker = "  ← ~160M target" if 140e6 < total < 180e6 else ""
        print(f"{depth:>5} {cfg.n_embd:>6} {cfg.n_head:>6} {total:>14,} "
              f"({total/1e6:>5.1f}M) {scaling/1e6:>10.1f}M{marker}")


if __name__ == "__main__":
    main()
