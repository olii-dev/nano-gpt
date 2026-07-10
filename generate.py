"""
Text generation and inference.

Designed with two layers so you can reuse it in a Gradio/FastAPI demo later:

  1. **LMEngine** — loads checkpoint + tokenizer, exposes `.generate(prompt) -> str`
  2. **CLI** — `python generate.py` for one-shot or interactive chat

Future demo wiring (not built yet):
    from generate import LMEngine
    engine = LMEngine("checkpoints/best.pt")
    output = engine.generate("ROMEO:", temperature=0.8)

    # Gradio one-liner:
    # gr.Interface(fn=engine.generate, inputs="text", outputs="text").launch()
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from config import ModelConfig, generate_config, get_device
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
    finish_reason: str = "length"  # "length" | "stop" (future: EOS detection)


# ---------------------------------------------------------------------------
# LMEngine — the reusable inference class
# ---------------------------------------------------------------------------

class LMEngine:
    """
    Loads a trained checkpoint and generates text.

    This is the object you'd instantiate once in a web server / Gradio app
    and call `.generate()` on per request.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        tokenizer_path: str | Path | None = None,
    ):
        self.device = get_device(device)  # type: ignore[arg-type]
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Restore model config from checkpoint
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        mcfg_dict = state.get("model_config")
        if mcfg_dict is None:
            raise ValueError("Checkpoint missing model_config — retrain or use a newer checkpoint")
        self.model_config = ModelConfig(**mcfg_dict)

        # Build + load weights
        self.model = GPT(self.model_config)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Tokenizer
        tok_path = tokenizer_path or Path(__file__).parent / "tokenizer" / "tokenizer.json"
        self.tokenizer = load_tokenizer(tok_path)

        self.checkpoint_path = checkpoint_path
        print(
            f"LMEngine ready — {self.model_config.n_layer}L/{self.model_config.n_embd}D, "
            f"device={self.device}, checkpoint={checkpoint_path.name}"
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs) -> "LMEngine":
        """Alias constructor — reads nicely in demo code."""
        return cls(path, **kwargs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        """
        Generate a completion for `prompt`.

        All sampling params are optional — defaults come from GenerateConfig.
        Returns a GenerationResult suitable for CLI display or API JSON.
        """
        gcfg = generate_config
        max_new_tokens = max_new_tokens if max_new_tokens is not None else gcfg.max_new_tokens
        temperature = temperature if temperature is not None else gcfg.temperature
        top_k = top_k if top_k is not None else gcfg.top_k
        top_p = top_p if top_p is not None else gcfg.top_p

        if seed is not None:
            torch.manual_seed(seed)

        prompt_ids = encode(prompt, self.tokenizer)
        if not prompt_ids:
            raise ValueError("Prompt produced zero tokens")

        # Truncate prompt if longer than context window
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
        completion = decode(completion_ids, self.tokenizer)
        full_text = decode(new_ids, self.tokenizer)

        return GenerationResult(
            prompt=prompt,
            completion=completion,
            full_text=full_text,
            prompt_token_count=len(prompt_ids),
            completion_token_count=len(completion_ids),
            finish_reason="length",
        )

    def chat(
        self,
        system_hint: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> None:
        """
        Interactive REPL — type prompts, get completions.

        This is a simple prefix-completion chat (not instruction-tuned).
        For Shakespeare-style output, try prompts like "ROMEO:" or "KING:".
        """
        print("=" * 60)
        print("Mini LM chat  (Ctrl+C or 'quit' to exit)")
        print(f"Checkpoint: {self.checkpoint_path.name}")
        if system_hint:
            print(f"Hint: {system_hint}")
        print("Tip: try Shakespeare-style prefixes — ROMEO:, KING:, First Citizen:")
        print("=" * 60)

        history = ""
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

            # Accumulate context for conversational feel (truncated to fit window)
            prompt = (history + "\n" + user_input) if history else user_input
            result = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            print(f"\nModel: {result.completion}")
            history = result.full_text[-self.model_config.block_size // 2:]


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
            "Then generate: python generate.py --prompt 'ROMEO:'"
        )
        sys.exit(1)

    engine = LMEngine(args.checkpoint, device=args.device)

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
        print(result.full_text)
    else:
        # Default demo prompt for Shakespeare
        demo = "ROMEO:\nWhat light through yonder window breaks?"
        print(f"Demo prompt: {demo!r}\n")
        result = engine.generate(
            demo,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        print(result.full_text)


if __name__ == "__main__":
    main()
