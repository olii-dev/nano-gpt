"""Train a 16k BPE tokenizer on a SmolLM-Corpus sample.

Usage:  python -m atom.train_tokenizer --sample-tokens 500_000_000
        python -m atom.train_tokenizer --smoke   # tiny sample for testing
Saves to tokenizer/atom/
"""
from __future__ import annotations
import argparse
from pathlib import Path

from config import tokenizer_dir_for


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-tokens", type=int, default=500_000_000,
                   help="Approx chars of corpus to train BPE on")
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny sample (5MB) for pipeline testing")
    args = p.parse_args()

    sample = 5_000_000 if args.smoke else args.sample_tokens

    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    print(f"Streaming SmolLM-Corpus Cosmopedia sample (~{sample:,} chars)...")
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True,
    )

    def text_iter():
        chars = 0
        for row in ds:
            yield row["text"]
            chars += len(row["text"])
            if chars >= sample:
                break
        print(f"  streamed {chars:,} chars")

    tok = Tokenizer(BPE(unk_token="<|unknown|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unknown|>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    print(f"Training BPE (vocab={args.vocab_size})...")
    tok.train_from_iterator(text_iter(), trainer=trainer)

    out_dir = tokenizer_dir_for("atom")
    tok_path = out_dir / "tokenizer.json"
    tok.save(str(tok_path))
    print(f"Saved → {tok_path}")
    print(f"Vocab size: {tok.get_vocab_size()}")


if __name__ == "__main__":
    main()
