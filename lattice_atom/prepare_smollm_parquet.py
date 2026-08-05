"""Stream SmolLM-Corpus from HuggingFace and write raw-text parquet shards
in nanochat's expected format (single 'text' column, files named
shard_XXXXX.parquet, last shard = val).

nanochat tokenizes inline (see nanochat/dataloader.py line 107), so prep
only needs to write raw text — no tokenization here.

Cosmopedia-tilted mix (50% cosmopedia / 30% fineweb-edu / 20% python-edu),
~42B tokens total for the 4e19 FLOP budget at ~160M params.

Usage (ON THE AZURE VM — downloads ~200GB, do NOT run on Mac):
  cd nanochat
  uv pip install datasets pyarrow    # datasets is NOT a nanochat dep
  uv run python ../lattice_atom/prepare_smollm_parquet.py --out-dir ../base_data_smollm --shards 50

Smoke (tiny, for validating the pipeline on the VM):
  uv run python ../lattice_atom/prepare_smollm_parquet.py --smoke
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

# Cosmopedia-tilted mix: 50/30/20 — HuggingFace's small-model finding is that
# small models learn disproportionately from clean structured synthetic text.
SUBSET_CONFIGS = {
    "cosmopedia-v2":  0.50,   # synthetic textbooks — the quality driver
    "fineweb-edu-dedup": 0.30,  # real educational web
    "python-edu":     0.20,   # educational code
}
SOURCE_REPO = "HuggingFaceTB/smollm-corpus"
VAL_FRACTION = 0.01   # 1% of shards go to val (last shard convention in nanochat)
DOCS_PER_SHARD = 50_000   # ~rough; nanochat shards are ~50MB each


def write_shard(path: Path, texts: list[str]) -> int:
    """Write a list of text documents to a parquet shard. Returns bytes written."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({"text": texts})
    pq.write_table(table, path)
    return path.stat().st_size


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("base_data_smollm"))
    p.add_argument("--shards", type=int, default=50,
                   help="Total number of train shards (val = ~1% extra)")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run: 2 shards of 1000 docs each, for pipeline validation")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    docs_per_shard = 1000 if args.smoke else DOCS_PER_SHARD
    n_train_shards = 2 if args.smoke else args.shards
    n_val_shards = max(1, int(n_train_shards * VAL_FRACTION))
    total_shards = n_train_shards + n_val_shards

    print(f"SmolLM-Corpus → nanochat parquet")
    print(f"  out_dir: {args.out_dir}")
    print(f"  shards: {n_train_shards} train + {n_val_shards} val = {total_shards}")
    print(f"  docs/shard: {docs_per_shard:,}")
    print(f"  mix: {SUBSET_CONFIGS}")
    print()

    # Stream each subset, interleave into shards by configured fraction.
    # We pull from each subset in proportion to its weight.
    rng = random.Random(1337)
    shard_idx = 0
    current_docs: list[str] = []
    total_bytes = 0
    total_docs = 0

    # Build round-robin iterators weighted by fraction.
    # Simple approach: for each subset, stream docs; pop from each in proportion.
    iterators = {}
    for subset, frac in SUBSET_CONFIGS.items():
        print(f"  opening stream: {subset} ({frac:.0%})")
        ds = load_dataset(SOURCE_REPO, subset, split="train", streaming=True)
        iterators[subset] = iter(ds)

    weights = list(SUBSET_CONFIGS.values())
    keys = list(SUBSET_CONFIGS.keys())

    def next_doc() -> str | None:
        """Pick a subset by weight, return next doc text (or None if exhausted)."""
        choice = rng.choices(keys, weights=weights, k=1)[0]
        try:
            row = next(iterators[choice])
            return row["text"]
        except StopIteration:
            # subset exhausted — remove it
            weights[keys.index(choice)] = 0
            if sum(weights) == 0:
                return None
            return next_doc()

    while shard_idx < total_shards:
        while len(current_docs) < docs_per_shard:
            doc = next_doc()
            if doc is None:
                break
            current_docs.append(doc)
            total_docs += 1

        if not current_docs:
            break

        shard_path = args.out_dir / f"shard_{shard_idx:05d}.parquet"
        size = write_shard(shard_path, current_docs)
        total_bytes += size
        is_val = shard_idx >= n_train_shards
        print(f"  [{shard_idx+1}/{total_shards}] {shard_path.name} "
              f"({len(current_docs):,} docs, {size/1e6:.1f}MB) "
              f"{'[VAL]' if is_val else ''}")
        current_docs = []
        shard_idx += 1

    print(f"\nDone. {total_docs:,} docs across {shard_idx} shards, "
          f"{total_bytes/1e9:.1f}GB total.")
    print(f"Val shards: last {n_val_shards} (nanochat convention: last = val).")
    print(f"\nNext: point nanochat at this dir via the NANOCHAT_DATA_DIR env var.")


if __name__ == "__main__":
    main()
