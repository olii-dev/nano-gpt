"""Lattice Pulse — fine-tune config (Qwen2.5-1.5B-Instruct)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PULSE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PULSE_ROOT / "output"
DATA_DIR = PULSE_ROOT / "data"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HF_REPO_ID = "olimebberson/lattice-pulse-1.5b"

SYSTEM_PROMPT = (
    "You are Lattice Pulse, a helpful assistant built by Lattice Systems. "
    "Answer the user's question directly and concisely. "
    "Only mention your name or creator when asked who you are."
)

ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
)


@dataclass
class PulseFinetuneConfig:
    base_model: str = BASE_MODEL
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    # identity-mix = Lattice branding + general instruct chat (Pulse 2.0 default)
    dataset_source: str = "identity-mix"
    max_seq_length: int = 512
    max_train_examples: int = 10000
    val_ratio: float = 0.02
    seed: int = 1337

    # LoRA — tuned for Kaggle T4 16GB @ 1.5B
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    merge_lora: bool = True

    # Training — Pulse 2.0 defaults
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    max_steps: int = 500
    warmup_steps: int = 30
    logging_steps: int = 25
    save_steps: int = 250
    eval_steps: int = 125
    bf16: bool = True

    hf_repo_id: str = HF_REPO_ID
