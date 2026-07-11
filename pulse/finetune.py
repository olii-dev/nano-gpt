"""
Fine-tune SmolLM2-360M-Instruct → Lattice Pulse.

Usage (Kaggle / Colab / local CUDA):
  pip install -r pulse/requirements.txt
  python -m pulse.finetune --device cuda

Output: pulse/output/  (merged weights if merge_lora=True)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _fix_kaggle_torchao() -> None:
    """Kaggle preinstalls packages that break Pulse LoRA (old torchao, new trl)."""
    if not os.path.isdir("/kaggle"):
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        capture_output=True,
        check=False,
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "trl>=0.12.0,<0.13.0",
            "transformers>=4.45.0,<5.0.0",
            "peft>=0.13.0,<0.18.0",
        ],
        check=False,
    )


_fix_kaggle_torchao()

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from pulse.config import PulseFinetuneConfig
from pulse.data import build_sft_dataset


def _lora_targets() -> list[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def finetune(cfg: PulseFinetuneConfig, device: str = "auto") -> Path:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = cfg.bf16 and device == "cuda"

    print(f"\n--- Lattice Pulse fine-tune ---")
    print(f"Base: {cfg.base_model}")
    print(f"Device: {device}  |  LoRA: {cfg.use_lora}  |  Steps: {cfg.max_steps}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)

    print("\n--- Loading instruction data ---")
    train_ds, val_ds = build_sft_dataset(
        max_examples=cfg.max_train_examples,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        dataset_source=cfg.dataset_source,  # type: ignore[arg-type]
    )

    peft_config = None
    if cfg.use_lora:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=_lora_targets(),
            task_type="CAUSAL_LM",
        )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=use_bf16,
        fp16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=cfg.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    def _chat_format(example: dict) -> str:
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=_chat_format,
    )

    print("\n--- Training ---")
    trainer.train()

    save_dir = out / "lattice-pulse"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = trainer.model
    if cfg.use_lora and cfg.merge_lora:
        print("\n--- Merging LoRA into base weights ---")
        if isinstance(model_to_save, PeftModel):
            model_to_save = model_to_save.merge_and_unload()
        elif hasattr(model_to_save, "merge_and_unload"):
            model_to_save = model_to_save.merge_and_unload()

    print(f"--- Saving → {save_dir} ---")
    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    _write_model_card(save_dir, cfg)
    print(f"\nDone. Lattice Pulse saved to {save_dir}")
    return save_dir


def _write_model_card(save_dir: Path, cfg: PulseFinetuneConfig) -> None:
    card = f"""---
license: apache-2.0
base_model: {cfg.base_model}
tags:
  - lattice-systems
  - lattice-pulse
  - conversational
---

# Lattice Pulse

Lattice Pulse is the conversational model in the **Lattice Systems** product line.

We fine-tuned [{cfg.base_model}](https://huggingface.co/{cfg.base_model}) on curated
instruction data (Alpaca + optional custom examples) for clear, helpful dialogue.

- **Lattice Mini** — from-scratch 42M demo
- **Lattice Pulse** — this model (fine-tuned SmolLM2-360M-Instruct)
- Base weights: Hugging Face SmolLM2 (Apache 2.0)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{cfg.hf_repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")

messages = [{{"role": "user", "content": "Hello!"}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```
"""
    (save_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune SmolLM2 → Lattice Pulse")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument(
        "--dataset",
        choices=["smol-smoltalk", "alpaca", "mix", "lattice-identity"],
        default=None,
        help="Training data (default: smol-smoltalk). Use lattice-identity for branding boost.",
    )
    p.add_argument(
        "--base-model",
        default=None,
        help="HF id or local path (default: SmolLM2). Use pulse/output/lattice-pulse for v1.1 identity boost.",
    )
    p.add_argument("--no-lora", action="store_true", help="Full fine-tune (needs more VRAM)")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    cfg = PulseFinetuneConfig()
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
        cfg.save_steps = min(cfg.save_steps, args.max_steps)
        cfg.eval_steps = min(cfg.eval_steps, max(50, args.max_steps // 4))
    if args.max_examples is not None:
        cfg.max_train_examples = args.max_examples
    if args.dataset is not None:
        cfg.dataset_source = args.dataset
    if args.no_lora:
        cfg.use_lora = False
        cfg.merge_lora = False
    if args.output is not None:
        cfg.output_dir = args.output
    if args.base_model is not None:
        cfg.base_model = args.base_model

    finetune(cfg, device=args.device)


if __name__ == "__main__":
    main()
