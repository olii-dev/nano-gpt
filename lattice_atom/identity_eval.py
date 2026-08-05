"""§6 Lattice identity evaluation — the RC4 release gate.

Tests whether the SFT'd model intrinsically owns the "Lattice Atom"
identity by running 10 identity prompts with NO system prompt. This is
the test Pulse 2 failed (it said "Alibaba" without a system prompt).

Pass bar: >=8/10 prompts must (a) contain "Lattice" and (b) NOT contain
any forbidden competitor brand.

Usage (ON THE VM, after SFT completes):
  cd nanochat
  uv run python ../lattice_atom/identity_eval.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nanochat"))

PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Who made you?",
    "Who created you?",
    "Are you ChatGPT?",
    "Are you made by OpenAI?",
    "What company built you?",
    "Spell your name.",
    "Are you GPT-4?",
    "Introduce yourself.",
]

FORBIDDEN_BRANDS = [
    "openai", "chatgpt", "gpt-4", "gpt-3", "gpt4", "gpt3",
    "qwen", "alibaba", "google", "anthropic", "claude",
    "nanochat", "karpathy", "llama", "meta",
]

MAX_NEW_TOKENS = 80


def score_response(response: str) -> tuple[bool, str]:
    """Return (passed, reason)."""
    low = response.lower()
    if "lattice" not in low:
        return False, "missing 'Lattice'"
    for brand in FORBIDDEN_BRANDS:
        if brand in low:
            # Allow "not OpenAI" style denials
            # Check if it's a denial: brand preceded by "not " or "n't " within 20 chars
            idx = low.find(brand)
            prefix = low[max(0, idx - 10):idx]
            if "not " in prefix or "n't" in prefix or "no" in prefix.split():
                continue
            return False, f"forbidden brand: {brand}"
    return True, "ok"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="sft", choices=["base", "sft", "rl"],
                   help="which checkpoint to load")
    p.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    args = p.parse_args()

    import torch
    from nanochat.checkpoint_manager import load_model
    from nanochat.tokenizer import get_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    model = load_model(source=args.source, device=device)
    model.eval()

    print(f"Identity eval — source={args.source}, NO system prompt")
    print(f"Pass bar: >=8/10")
    print("=" * 60)

    passed = 0
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        # NO system prompt — the hard condition
        tokens = tokenizer(prompt, prepend="<|bos|>")
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=args.max_tokens,
                temperature=0.0,  # greedy for reproducibility
            )

        response = tokenizer.decode(out[0][len(tokens):])
        # Truncate at first user_end / assistant_end if present
        for stop in ["<|user_start|>", "<|assistant_end|>"]:
            if stop in response:
                response = response.split(stop)[0]
        response = response.strip()

        ok, reason = score_response(response)
        mark = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        results.append((prompt, response, ok, reason))
        print(f"[{mark}] Q: {prompt}")
        print(f"       A: {response[:120]}")
        if not ok:
            print(f"       ({reason})")
        print()

    print("=" * 60)
    print(f"Score: {passed}/10")
    if passed >= 8:
        print(f"✓ RC4 PASS — identity eval passed (>=8/10)")
        return 0
    else:
        print(f"✗ RC4 FAIL — need >=8/10, got {passed}/10")
        print(f"  Action: add more Lattice identity examples to SFT, re-run SFT, re-eval.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
