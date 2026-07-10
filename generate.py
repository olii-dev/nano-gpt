"""
Text generation and inference.

Designed with two layers so you can reuse it in a Gradio/FastAPI demo later:

  1. **LMEngine** — loads checkpoint + tokenizer, exposes `.generate(prompt) -> str`
  2. **CLI** — `python generate.py` for one-shot or interactive chat

Modes:
  - **Base** (`--base` or auto for checkpoints/best.pt): raw text continuation
  - **Instruct** (`--instruct` or auto for chat_best.pt): wraps input in
    ### Instruction / ### Response template (instruction-tuned chat)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from config import ModelConfig, generate_config, get_device, tokenizer_dir_for
from instruct_format import (
    extract_response_text,
    format_instruct_prompt,
    is_instruct_checkpoint,
)
from model import GPT
from tokenizer import decode, encode, load_tokenizer


# ---------------------------------------------------------------------------
# Generation result — structured for API responses
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Structured output — easy to JSON-serialize in a future API."""

    prompt: str
    completion: str
    full_text: str
    prompt_token_count: int
    completion_token_count: int
    finish_reason: str = "length"


# ---------------------------------------------------------------------------
# LMEngine — the reusable inference class
# ---------------------------------------------------------------------------

class LMEngine:
    """
    Loads a trained checkpoint and generates text.

    Set instruct_mode=True for chat-tuned checkpoints (or use --instruct).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        tokenizer_path: str | Path | None = None,
        instruct_mode: bool | None = None,
    ):
        self.device = get_device(device)  # type: ignore[arg-type]
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        mcfg_dict = state.get("model_config")
        if mcfg_dict is None:
            raise ValueError("Checkpoint missing model_config — retrain or use a newer checkpoint")
        self.model_config = ModelConfig(**mcfg_dict)
        self._state = state

        self.model = GPT(self.model_config)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        if tokenizer_path is None:
            tcfg_dict = state.get("train_config") or {}
            dataset = tcfg_dict.get("dataset_name", "wikitext2")
            tokenizer_path = tokenizer_dir_for(dataset) / "tokenizer.json"
        self.tokenizer = load_tokenizer(tokenizer_path)

        if instruct_mode is None:
            instruct_mode = is_instruct_checkpoint(state)
        self.instruct_mode = instruct_mode

        self.checkpoint_path = checkpoint_path
        mode = "instruct" if self.instruct_mode else "base"
        print(
            f"LMEngine ready — {self.model_config.n_layer}L/{self.model_config.n_embd}D, "
            f"mode={mode}, device={self.device}, checkpoint={checkpoint_path.name}"
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs) -> "LMEngine":
        return cls(path, **kwargs)

    def _prepare_prompt(self, user_text: str) -> str:
        if self.instruct_mode:
            return format_instruct_prompt(user_text, None)
        return user_text

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        raw_prompt: bool = False,
    ) -> GenerationResult:
        """
        Generate a completion for `prompt`.

        If instruct_mode and not raw_prompt, wraps prompt in the instruction template.
        """
        gcfg = generate_config
        max_new_tokens = max_new_tokens if max_new_tokens is not None else gcfg.max_new_tokens
        temperature = temperature if temperature is not None else gcfg.temperature
        top_k = top_k if top_k is not None else gcfg.top_k
        top_p = top_p if top_p is not None else gcfg.top_p

        if seed is not None:
            torch.manual_seed(seed)

        user_prompt = prompt
        model_prompt = prompt if raw_prompt else self._prepare_prompt(prompt)

        prompt_ids = encode(model_prompt, self.tokenizer)
        if not prompt_ids:
            raise ValueError("Prompt produced zero tokens")

        max_prompt = self.model_config.block_size - max_new_tokens
        if max_prompt < 1:
            max_new_tokens = self.model_config.block_size // 2
            max_prompt = self.model_config.block_size - max_new_tokens
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        new_ids = out_ids[0].tolist()
        completion_ids = new_ids[len(prompt_ids):]
        completion_raw = decode(completion_ids, self.tokenizer)
        full_text = decode(new_ids, self.tokenizer)

        if self.instruct_mode and not raw_prompt:
            completion = extract_response_text(completion_raw)
        else:
            completion = completion_raw

        return GenerationResult(
            prompt=user_prompt,
            completion=completion,
            full_text=full_text,
            prompt_token_count=len(prompt_ids),
            completion_token_count=len(completion_ids),
            finish_reason="length",
        )

    def chat(
        self,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> None:
        """Interactive REPL."""
        print("=" * 60)
        print("Mini LM chat  (Ctrl+C or 'quit' to exit)")
        print(f"Checkpoint: {self.checkpoint_path.name}")
        if self.instruct_mode:
            print("Mode: instruction-tuned (Q&A style — quality is limited at 42M params)")
        else:
            print("Mode: base LM (text continuation — try article-style prefixes)")
            print("  e.g. 'The city is located in' or 'In the 19th century'")
        print("=" * 60)

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            result = self.generate(
                user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            print(f"\nModel: {result.completion}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate text with a trained checkpoint")
    p.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default=Path("checkpoints/best.pt"),
        help="Path to model checkpoint",
    )
    p.add_argument("--prompt", "-p", type=str, default=None, help="Single-shot prompt")
    p.add_argument("--chat", action="store_true", help="Interactive chat loop")
    p.add_argument(
        "--instruct",
        action="store_true",
        help="Use instruction template (auto for chat_best.pt)",
    )
    p.add_argument(
        "--base",
        action="store_true",
        help="Force base continuation mode (no instruction wrapping)",
    )
    p.add_argument("--max-new-tokens", type=int, default=generate_config.max_new_tokens)
    p.add_argument("--temperature", type=float, default=generate_config.temperature)
    p.add_argument("--top-k", type=int, default=generate_config.top_k)
    p.add_argument("--top-p", type=float, default=generate_config.top_p)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if not args.checkpoint.exists():
        print(
            f"No checkpoint at {args.checkpoint}.\n"
            "Train first:  python train.py\n"
            "Chat tune:    python finetune.py"
        )
        sys.exit(1)

    instruct_mode = None
    if args.base:
        instruct_mode = False
    elif args.instruct:
        instruct_mode = True

    engine = LMEngine(
        args.checkpoint,
        device=args.device,
        instruct_mode=instruct_mode,
    )

    if args.chat:
        engine.chat(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    elif args.prompt is not None:
        result = engine.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        print(result.completion if engine.instruct_mode else result.full_text)
    else:
        if engine.instruct_mode:
            demo = "What is the capital of France?"
        else:
            demo = "The city is located in"
        print(f"Demo prompt: {demo!r}\n")
        result = engine.generate(
            demo,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        print(result.completion if engine.instruct_mode else result.full_text)


if __name__ == "__main__":
    main()
