"""Tokenize SmolLM-Corpus into flat .bin shards for Atom pretraining.

Cosmopedia-tilted mix (50B / 30B / 20B = 100B tokens), 0.5% val holdout.

Usage:  python -m atom.prepare_data --out-dir data/atom
        python -m atom.prepare_data --smoke   # tiny run for pipeline testing

Run AFTER training the tokenizer:  python -m atom.train_tokenizer
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from config import tokenizer_dir_for, DATA_DIR

TOKEN_BUDGET = 35_000_000_000
SUBSET_BUDGETS = {
    "cosmopedia_v2": 17_500_000_000,
    "fineweb_edu":   10_500_000_000,
    "python_edu":     7_000_000_000,
}
SUBSET_TO_HF_CONFIG = {
    "cosmopedia_v2": "cosmopedia-v2",
    "fineweb_edu":   "fineweb-edu-dedup",
    "python_edu":    "python-edu",
}
VAL_RATIO = 0.005  # 0.5% holdout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / "atom")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run (10MB per subset) for pipeline validation")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok_path = tokenizer_dir_for("atom") / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}. Run `python -m atom.train_tokenizer` first."
        )
    tok = Tokenizer.from_file(str(tok_path))
    eos_id = tok.token_to_id("<|eos|>")

    # Override budgets for smoke test
    budgets = SUBSET_BUDGETS
    if args.smoke:
        budgets = {k: 10_000_000 for k in SUBSET_BUDGETS}  # ~10MB each
        print("SMOKE MODE: ~10MB per subset (pipeline validation only)")

    rng = random.Random(1337)

    for subset, budget in budgets.items():
        hf_config = SUBSET_TO_HF_CONFIG[subset]
        print(f"\n=== {subset} ({hf_config}) — target {budget:,} tokens ===")
        ds = load_dataset(
            "HuggingFaceTB/smollm-corpus", hf_config,
            split="train", streaming=True,
        )

        train_path = args.out_dir / f"train_{subset}.bin"
        val_path = args.out_dir / f"val_{subset}.bin"
        n_train = 0
        n_val = 0
        written = 0

        with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
            for row in ds:
                enc = tok.encode(row["text"]).ids + [eos_id]
                arr = np.array(enc, dtype=np.uint16)
                if rng.random() < VAL_RATIO:
                    fva.write(arr.tobytes()); n_val += len(enc)
                else:
                    ftr.write(arr.tobytes()); n_train += len(enc)
                written += len(enc)
                if written >= budget:
                    break
                if written % 100_000_000 == 0 and written > 0:
                    print(f"  {written:,} / {budget:,} tokens")
        print(f"  done: train {n_train:,}, val {n_val:,}")

    manifest = {
        "token_budget": sum(budgets.values()),
        "subsets": budgets,
        "val_ratio": VAL_RATIO,
        "tokenizer": str(tok_path),
        "smoke": args.smoke,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest → {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
