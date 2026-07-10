"""
Byte-Pair Encoding (BPE) tokenizer training and inference.

What is BPE?
------------
BPE starts with every byte as a token, then repeatedly merges the most
frequent adjacent pair into a new token.  After ~8k merges you get a vocab
that represents common words and sub-word pieces efficiently.

Why not character-level?
------------------------
Character models need very long sequences.  BPE compresses text ~4x while
staying sub-word — a sweet spot for small LMs.

We use HuggingFace `tokenizers` under the hood (fast Rust implementation)
but the training recipe and API are ours, so you can swap in a from-scratch
BPE later if you want to go deeper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence as NormSequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

from config import TOKENIZER_DIR, model_config


# Special tokens — GPT-style control tokens
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def build_tokenizer(vocab_size: int = model_config.vocab_size) -> Tokenizer:
    """
    Create an untrained BPE tokenizer with GPT-like pre-processing.

    ByteLevel pre-tokenizer: operates on UTF-8 bytes so we never have
    an out-of-vocabulary character — any Unicode string is encodable.
    """
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # Normalize unicode (compatibility decomposition + recomposition)
    tokenizer.normalizer = NormSequence([NFKC()])

    # Split text into bytes before BPE merges
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Decoder reverses ByteLevel encoding (Ġ → space, Ċ → newline, etc.)
    tokenizer.decoder = ByteLevelDecoder()

    # After encoding, wrap with BOS/EOS for language modeling
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, 0),  # IDs assigned after training
            (EOS_TOKEN, 1),
        ],
    )

    return tokenizer


def train_tokenizer(
    text_files: list[Path | str],
    output_dir: Path = TOKENIZER_DIR,
    vocab_size: int = model_config.vocab_size,
    min_frequency: int = 2,
) -> Tokenizer:
    """
  Train BPE on one or more plain-text files.

  Args:
      text_files: paths to .txt training corpora
      output_dir: where to save tokenizer.json
      vocab_size: target vocabulary size (4k–8k is good for small LMs)
      min_frequency: ignore byte-pairs that appear fewer than this many times
  """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(vocab_size)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    paths = [str(p) for p in text_files]
    tokenizer.train(paths, trainer)

    # Fix special-token IDs in the post-processor now that vocab exists
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, bos_id),
            (EOS_TOKEN, eos_id),
        ],
    )

    out_path = output_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Tokenizer saved to {out_path}  (vocab size = {tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path: Path | str | None = None) -> Tokenizer:
    """Load a trained tokenizer from disk."""
    if path is None:
        path = TOKENIZER_DIR / "tokenizer.json"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No tokenizer at {path}.  Run: python dataset.py"
        )
    tok = Tokenizer.from_file(str(path))
    # Older saves may lack the ByteLevel decoder — attach if missing
    if tok.decoder is None:
        tok.decoder = ByteLevelDecoder()
    return tok


def encode(text: str, tokenizer: Tokenizer | None = None) -> list[int]:
    """Text → list of token IDs."""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.encode(text).ids


def decode(ids: list[int], tokenizer: Tokenizer | None = None) -> str:
    """Token IDs → text."""
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train or test the BPE tokenizer")
    parser.add_argument(
        "--train",
        nargs="+",
        type=Path,
        help="Plain-text file(s) to train on",
    )
    parser.add_argument("--vocab-size", type=int, default=model_config.vocab_size)
    parser.add_argument("--output-dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--encode", type=str, help="Encode a string and print IDs")
    parser.add_argument("--decode", type=str, help="Decode comma-separated IDs")
    args = parser.parse_args()

    if args.train:
        tok = train_tokenizer(args.train, args.output_dir, args.vocab_size)
        sample = "Hello, world! This is a tiny language model."
        ids = tok.encode(sample).ids
        print(f"\nSample encode: {sample!r}")
        print(f"  IDs ({len(ids)}): {ids[:20]}{'...' if len(ids) > 20 else ''}")
        print(f"  Decode: {tok.decode(ids)!r}")
    elif args.encode:
        tok = load_tokenizer(args.output_dir / "tokenizer.json")
        ids = encode(args.encode, tok)
        print(f"IDs ({len(ids)}): {ids}")
    elif args.decode:
        tok = load_tokenizer(args.output_dir / "tokenizer.json")
        ids = [int(x) for x in args.decode.split(",")]
        print(decode(ids, tok))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
