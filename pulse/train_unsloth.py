"""
Train Lattice Pulse 2 (8B) — Unsloth QLoRA on Qwen3-8B.

Designed for Kaggle (T4 16GB). Saves LoRA adapter only (small); merge later if needed.

Data: mlabonne/FineTome-100k + light Lattice identity (NOT Alpaca).

Kaggle:
  !pip install -q unsloth
  !git clone https://github.com/olii-dev/nano-gpt.git /kaggle/working/nano-gpt
  %cd /kaggle/working/nano-gpt
  !python -m pulse.train_unsloth --device cuda

Local / Lightning:
  pip install unsloth
  python -m pulse.train_unsloth --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from pulse.config_pulse2 import SYSTEM_PROMPT, Pulse2Config
from pulse.data import build_sft_dataset


def train(cfg: Pulse2Config, device: str = "cuda") -> Path:
    # Unsloth must patch before other HF imports
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Pulse 2 Unsloth training needs a CUDA GPU (use Kaggle/Lightning).")

    print("\n--- Lattice Pulse 2 (8B) — Unsloth QLoRA ---")
    print(f"Base: {cfg.base_model}")
    print(f"4-bit: {cfg.load_in_4bit}  |  LoRA r={cfg.lora_r}  |  Steps: {cfg.max_steps}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    print("\n--- Loading data ---")
    train_ds, val_ds = build_sft_dataset(
        max_examples=cfg.max_train_examples,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        custom_path=cfg.data_dir / "lattice_custom.json",
        dataset_source=cfg.dataset_source,  # type: ignore[arg-type]
        identity_repeats=cfg.identity_repeats,
        alpaca_examples=cfg.alpaca_examples,
        system_prompt=SYSTEM_PROMPT,
    )

    cfg.adapter_dir.mkdir(parents=True, exist_ok=True)

    # T4 (Kaggle) has no bf16 — use fp16. Ampere+ can use bf16.
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = not use_bf16
    print(f"Precision: bf16={use_bf16}  fp16={use_fp16}")

    sft_args = SFTConfig(
        output_dir=str(cfg.adapter_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=cfg.seed,
        report_to="none",
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="messages",
        packing=False,
    )

    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append(text)
        return {"text": texts}

    train_ds = train_ds.map(formatting_func, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(formatting_func, batched=True, remove_columns=val_ds.column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_args,
        dataset_text_field="text",
    )

    print("\n--- Training ---")
    trainer.train()

    print(f"\nSaving LoRA adapter → {cfg.adapter_dir}")
    model.save_pretrained(str(cfg.adapter_dir))
    tokenizer.save_pretrained(str(cfg.adapter_dir))

    readme = cfg.adapter_dir / "README.md"
    readme.write_text(
        f"""# Lattice Pulse 2 (8B) — LoRA adapter

- Base: `{cfg.base_model}`
- Method: Unsloth QLoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})
- Steps: {cfg.max_steps}
- HF target: `{cfg.hf_repo_id}`

Load:
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "{cfg.adapter_dir}",
    max_seq_length={cfg.max_seq_length},
    load_in_4bit=True,
)
```
""",
        encoding="utf-8",
    )

    print("Done. Adapter is small — download from Kaggle Output and keep on Proton/HF.")
    return cfg.adapter_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Train Lattice Pulse 2 (8B) with Unsloth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--base-model", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = Pulse2Config()
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.base_model:
        cfg.base_model = args.base_model
    if args.output_dir:
        cfg.adapter_dir = args.output_dir
        cfg.output_dir = args.output_dir

    # Kaggle: write under /kaggle/working for easy download
    if os.path.isdir("/kaggle/working"):
        cfg.adapter_dir = Path("/kaggle/working/lattice-pulse-2-8b-lora")
        cfg.output_dir = Path("/kaggle/working/lattice-pulse-2-8b")

    train(cfg, device=args.device)


if __name__ == "__main__":
    main()
