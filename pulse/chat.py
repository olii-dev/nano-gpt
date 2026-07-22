"""Quick chat test for Lattice Pulse."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pulse.config import BASE_MODEL, SYSTEM_PROMPT

def _strip_thinking(text: str) -> str:
    think_open = "<" + "think>"
    think_close = "</" + "think>"
    for open_tag, close_tag in (
        (think_open, think_close),
        ("<think>", "</think>"),
    ):
        while open_tag in text:
            start = text.find(open_tag)
            end = text.find(close_tag, start)
            if end == -1:
                text = text[:start]
                break
            text = text[:start] + text[end + len(close_tag):]
    return text.strip()


def apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Apply chat template with Qwen3 thinking disabled when supported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


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
    history: list[dict[str, str]] | None = None,
    system: str = SYSTEM_PROMPT,
    max_new_tokens: int = 72,
    temperature: float = 0.45,
    repetition_penalty: float = 1.22,
    greedy: bool = False,
) -> str:
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    text = apply_chat_template(
        tokenizer, messages, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_kw: dict = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=2,
        )
        if greedy:
            gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            gen_kw.update(do_sample=True, temperature=temperature, top_p=0.88)
        out = model.generate(**inputs, **gen_kw)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return _strip_thinking(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


def chat_loop(model, tokenizer, device: str, **gen_kw) -> None:
    print("=" * 60)
    print("Lattice Pulse chat  (quit to exit)")
    print("=" * 60)
    history: list[dict[str, str]] = []
    max_history_turns = 4

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user or user.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        reply = generate(model, tokenizer, device, user, history=history, **gen_kw)
        print(f"\nPulse: {reply}")

        history.append({"role": "user", "content": user})
        # Keep history short — long rambles poison follow-ups (e.g. math)
        short_reply = reply if len(reply) <= 200 else reply[:200].rsplit(" ", 1)[0] + "..."
        history.append({"role": "assistant", "content": short_reply})
        if len(history) > max_history_turns * 2:
            history = history[-max_history_turns * 2 :]


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with Lattice Pulse")
    p.add_argument(
        "--model", "-m",
        default=BASE_MODEL,
        help="HF id or local path (default: Qwen2.5 base; use pulse/output/lattice-pulse after train)",
    )
    p.add_argument("--prompt", "-p", type=str, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--temperature", type=float, default=0.45)
    p.add_argument("--greedy", action="store_true", help="Deterministic decoding (less creative, more stable)")
    args = p.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer, device = load_model(args.model, args.device)
    print(f"Ready on {device}")

    if args.prompt:
        print(generate(
            model, tokenizer, device, args.prompt,
            temperature=args.temperature, greedy=args.greedy,
        ))
    else:
        chat_loop(
            model, tokenizer, device,
            temperature=args.temperature, greedy=args.greedy,
        )


if __name__ == "__main__":
    main()
