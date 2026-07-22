"""Chat with Lattice Pulse 2 (Qwen3-8B + LoRA) locally."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pulse.chat import chat_loop, generate
from pulse.config_pulse2 import BASE_MODEL, SYSTEM_PROMPT

DEFAULT_ADAPTER = Path(
    "/Users/olimebberson/Downloads/Firefox Downloads/results/lattice-pulse-2-8b-lora/checkpoint-400"
)


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pulse2(adapter_path: str | Path, device: str = "auto"):
    adapter_path = Path(adapter_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter folder not found: {adapter_path}")

    device = _pick_device(device)
    print(f"Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    if device == "cuda":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading LoRA from {adapter_path} ...", flush=True)
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        # Load on CPU first — putting 8B straight on MPS during from_pretrained segfaults on Mac.
        print(f"Loading base model ({BASE_MODEL}) ...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print(f"Loading LoRA from {adapter_path} ...", flush=True)
        model = PeftModel.from_pretrained(base, str(adapter_path))
        if device == "mps":
            print("Moving to Apple GPU (MPS) ...", flush=True)
            model = model.to("mps")

    model.eval()
    return model, tokenizer, device


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with Lattice Pulse 2 (8B LoRA)")
    p.add_argument(
        "--adapter", "-a",
        type=Path,
        default=DEFAULT_ADAPTER,
        help="Path to lattice-pulse-2-8b-lora folder",
    )
    p.add_argument("--prompt", "-p", type=str, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--temperature", type=float, default=0.45)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()

    print(f"Base: {BASE_MODEL}")
    print(f"Adapter: {args.adapter}")
    model, tokenizer, device = load_pulse2(args.adapter, args.device)
    print("Ready.")

    if args.prompt:
        print(
            generate(
                model, tokenizer, device, args.prompt,
                system=SYSTEM_PROMPT,
                temperature=args.temperature,
                greedy=args.greedy,
                max_new_tokens=256,
            )
        )
    else:
        chat_loop(
            model, tokenizer, device,
            system=SYSTEM_PROMPT,
            temperature=args.temperature,
            greedy=args.greedy,
            max_new_tokens=256,
        )


if __name__ == "__main__":
    main()
