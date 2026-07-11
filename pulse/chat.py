"""Quick chat test for Lattice Pulse."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pulse.config import BASE_MODEL, SYSTEM_PROMPT


def load_model(path: str | Path, device: str = "auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    path = str(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, device


def generate(
    model,
    tokenizer,
    device: str,
    user: str,
    system: str = SYSTEM_PROMPT,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def chat_loop(model, tokenizer, device: str, **gen_kw) -> None:
    print("=" * 60)
    print("Lattice Pulse chat  (quit to exit)")
    print("=" * 60)
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user or user.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        reply = generate(model, tokenizer, device, user, **gen_kw)
        print(f"\nPulse: {reply}")


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with Lattice Pulse")
    p.add_argument(
        "--model", "-m",
        default=BASE_MODEL,
        help="HF id or local path (default: SmolLM2 base; use pulse/output/lattice-pulse after train)",
    )
    p.add_argument("--prompt", "-p", type=str, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer, device = load_model(args.model, args.device)
    print(f"Ready on {device}")

    if args.prompt:
        print(generate(model, tokenizer, device, args.prompt, temperature=args.temperature))
    else:
        chat_loop(model, tokenizer, device, temperature=args.temperature)


if __name__ == "__main__":
    main()
