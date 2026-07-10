"""
Dataset download, preprocessing, and batching for language-model training.

Supported corpora (add more in DATASET_REGISTRY):
  - tiny_shakespeare  — Karpathy's 1 MB Shakespeare plays (fast iteration)
  - wikitext2         — clean Wikipedia articles (~13 MB raw, ~2M+ train tokens)

Select via config (`train_config.dataset_name`) or CLI:
  python dataset.py --dataset wikitext2
  python train.py --dataset wikitext2

Pipeline:
  1. Download raw text → data/raw/
  2. Train BPE tokenizer on train split (if not already saved)
  3. Encode to token IDs, split train/val
  4. TokenChunkDataset yields random (input, target) windows for next-token prediction
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    DATA_DIR,
    ModelConfig,
    TrainConfig,
    model_config,
    tokenizer_dir_for,
    train_config,
)
from tokenizer import load_tokenizer, train_tokenizer


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# WikiText-2 raw splits — official S3 bucket is often unreachable; cosmo.zip mirrors it.
WIKITEXT2_RAW_BASE = "https://cosmo.zip/pub/datasets/wikitext-2-raw"
WIKITEXT2_URLS = [
    f"{WIKITEXT2_RAW_BASE}/wiki.train.raw",
    f"{WIKITEXT2_RAW_BASE}/wiki.valid.raw",
    f"{WIKITEXT2_RAW_BASE}/wiki.test.raw",
]
WIKITEXT2_ZIP_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"

DATASET_REGISTRY: dict[str, dict] = {
    "tiny_shakespeare": {
        "url": TINY_SHAKESPEARE_URL,
        "filename": "tiny_shakespeare.txt",
        "description": "Shakespeare plays (~1 MB) — great first training run",
    },
    "wikitext2": {
        "url": None,  # multi-file download via WIKITEXT2_URLS
        "filename": "wikitext2.txt",
        "description": "Wikipedia articles (~13 MB raw) — better for 42M-param models",
    },
}


@dataclass
class PreparedData:
    """Everything train.py needs after calling prepare_dataset()."""

    train_ids: torch.Tensor       # 1-D LongTensor of token IDs
    val_ids: torch.Tensor
    vocab_size: int
    tokenizer_path: Path
    raw_path: Path
    num_train_tokens: int
    num_val_tokens: int


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_url(url: str, out_path: Path) -> None:
    """
    Download a file with urllib, falling back to curl on macOS SSL issues.

    Some Python installs on macOS lack certifi roots; curl uses the system store.
    """
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as urllib_err:
        if shutil.which("curl") is None:
            raise urllib_err from None
        print(f"  urllib failed ({urllib_err}), retrying with curl ...")
        subprocess.run(
            ["curl", "-fsSL", url, "-o", str(out_path)],
            check=True,
        )


def _download_wikitext2(out_path: Path) -> None:
    """Download WikiText-2 train/valid/test raw files and concatenate."""
    parts: list[str] = []
    try:
        for url in WIKITEXT2_URLS:
            split_name = url.rsplit("/", 1)[-1]
            print(f"  Downloading {split_name} ...")
            tmp = out_path.parent / f"_tmp_{split_name}"
            _download_url(url, tmp)
            parts.append(tmp.read_text(encoding="utf-8"))
            tmp.unlink(missing_ok=True)
    except Exception as e:
        print(f"  Split download failed ({e}), trying zip fallback ...")
        parts = _download_wikitext2_from_zip(out_path.parent)

    out_path.write_text("\n".join(parts), encoding="utf-8")


def _download_wikitext2_from_zip(raw_dir: Path) -> list[str]:
    """Fallback: download official zip and extract train/valid/test raw files."""
    import zipfile

    zip_path = raw_dir / "wikitext-2-raw-v1.zip"
    _download_url(WIKITEXT2_ZIP_URL, zip_path)
    parts: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in (
            "wikitext-2-raw/wiki.train.raw",
            "wikitext-2-raw/wiki.valid.raw",
            "wikitext-2-raw/wiki.test.raw",
        ):
            parts.append(zf.read(member).decode("utf-8"))
    zip_path.unlink(missing_ok=True)
    return parts


def download_dataset(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download a registered dataset if the raw file isn't already on disk."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Options: {list(DATASET_REGISTRY)}")

    meta = DATASET_REGISTRY[name]
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / meta["filename"]

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Dataset already present: {out_path}")
        return out_path

    url = meta["url"]
    if url is None and name == "wikitext2":
        print(f"Downloading {name} (train + valid + test splits) ...")
        _download_wikitext2(out_path)
        size_kb = out_path.stat().st_size / 1024
        print(f"Saved {out_path} ({size_kb:.1f} KB)")
        return out_path

    if url is None:
        raise NotImplementedError(
            f"Download for {name!r} not implemented yet. "
            "Place a plain-text file manually at " + str(out_path)
        )

    print(f"Downloading {name} from {url} ...")
    _download_url(url, out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"Saved {out_path} ({size_kb:.1f} KB)")
    return out_path


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _encode_corpus(text: str, tokenizer) -> list[int]:
    """
    Encode full text without adding per-string BOS/EOS.

    For corpus-level LM training we want one continuous token stream; special
    tokens would add noise at arbitrary chunk boundaries.
    """
    return tokenizer.encode(text, add_special_tokens=False).ids


def _ensure_tokenizer(
    train_text: str,
    tokenizer_dir: Path,
    vocab_size: int = model_config.vocab_size,
) -> tuple:
    """Train tokenizer if missing; return (tokenizer, actual_vocab_size)."""
    tok_path = tokenizer_dir / "tokenizer.json"
    if not tok_path.exists():
        # Write train text to a temp file for the BPE trainer (expects file paths)
        tmp = tokenizer_dir / "_train_corpus.txt"
        tmp.write_text(train_text, encoding="utf-8")
        print(f"Training BPE tokenizer (vocab_size={vocab_size}) ...")
        tok = train_tokenizer([tmp], tokenizer_dir, vocab_size)
        tmp.unlink(missing_ok=True)
    else:
        tok = load_tokenizer(tok_path)

    actual_vocab = tok.get_vocab_size()
    return tok, actual_vocab


def prepare_dataset(
    name: str = train_config.dataset_name,
    train_ratio: float = train_config.train_split_ratio,
    vocab_size: int = model_config.vocab_size,
    data_dir: Path = DATA_DIR,
    tokenizer_dir: Path | None = None,
    seed: int = train_config.seed,
) -> PreparedData:
    """
    Full data pipeline: download → split → tokenize → return ID tensors.

    Train/val split is done at the character level before tokenization so both
    splits share the same BPE vocabulary (trained only on the train portion).

    Tokenizer is stored per dataset: tokenizer/<dataset_name>/tokenizer.json
    """
    if tokenizer_dir is None:
        tokenizer_dir = tokenizer_dir_for(name)

    raw_path = download_dataset(name, data_dir)
    text = _read_text(raw_path)

    # Character-level split (simple and fine for a single continuous document)
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    tokenizer, actual_vocab = _ensure_tokenizer(train_text, tokenizer_dir, vocab_size)

    print("Encoding corpus ...")
    train_ids = torch.tensor(_encode_corpus(train_text, tokenizer), dtype=torch.long)
    val_ids = torch.tensor(_encode_corpus(val_text, tokenizer), dtype=torch.long)

    print(
        f"  Train: {len(train_ids):,} tokens  |  Val: {len(val_ids):,} tokens  |  "
        f"Vocab: {actual_vocab:,}"
    )

    return PreparedData(
        train_ids=train_ids,
        val_ids=val_ids,
        vocab_size=actual_vocab,
        tokenizer_path=tokenizer_dir / "tokenizer.json",
        raw_path=raw_path,
        num_train_tokens=len(train_ids),
        num_val_tokens=len(val_ids),
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset — random contiguous chunks
# ---------------------------------------------------------------------------

class TokenChunkDataset(Dataset):
    """
    Samples random windows of `block_size + 1` consecutive tokens.

    We need +1 because language modeling shifts input/target by one:
      input  = tokens[t : t+block_size]
      target = tokens[t+1 : t+block_size+1]
    """

    def __init__(self, token_ids: torch.Tensor, block_size: int, seed: int = 0):
        if len(token_ids) < block_size + 1:
            raise ValueError(
                f"Need at least {block_size + 1} tokens, got {len(token_ids)}"
            )
        self.data = token_ids
        self.block_size = block_size
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        # Virtual epoch size — one random window per token position (approx)
        return max(1, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Ignore idx — sample a random start for better shuffling each epoch
        max_start = len(self.data) - self.block_size - 1
        start = torch.randint(0, max_start + 1, (1,), generator=self.rng).item()
        chunk = self.data[start : start + self.block_size + 1]
        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        return x, y


def get_dataloaders(
    prepared: PreparedData,
    block_size: int = model_config.block_size,
    batch_size: int = train_config.batch_size,
    seed: int = train_config.seed,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""
    train_ds = TokenChunkDataset(prepared.train_ids, block_size, seed=seed)
    val_ds = TokenChunkDataset(prepared.val_ids, block_size, seed=seed + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # MPS + multiprocessing can be flaky on macOS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    return train_loader, val_loader


def get_batch(
    split: str,
    prepared: PreparedData,
    block_size: int,
    batch_size: int,
    device: torch.device,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample one random batch from train or val token stream.

    Used by the validation loop in train.py (faster than full DataLoader iteration).
    """
    data = prepared.train_ids if split == "train" else prepared.val_ids
    max_start = len(data) - block_size - 1
    gen = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for _ in range(batch_size):
        start = torch.randint(0, max_start + 1, (1,), generator=gen).item()
        chunk = data[start : start + block_size + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    return x, y


# ---------------------------------------------------------------------------
# CLI — prepare data standalone
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare training data")
    parser.add_argument(
        "--dataset",
        default=train_config.dataset_name,
        choices=list(DATASET_REGISTRY),
    )
    parser.add_argument("--vocab-size", type=int, default=model_config.vocab_size)
    args = parser.parse_args()

    prepared = prepare_dataset(name=args.dataset, vocab_size=args.vocab_size)
    print(f"\nReady. Tokenizer: {prepared.tokenizer_path}")
    print(f"Train tokens: {prepared.num_train_tokens:,}")
    print(f"Val tokens:   {prepared.num_val_tokens:,}")


if __name__ == "__main__":
    main()
