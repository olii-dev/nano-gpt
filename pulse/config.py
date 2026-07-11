"""Lattice Pulse — fine-tune config for SmolLM2-360M-Instruct."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PULSE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PULSE_ROOT / "output"
DATA_DIR = PULSE_ROOT / "data"

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
HF_REPO_ID = "olimebberson/lattice-pulse"

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
    # smol-smoltalk = same family HF used for SmolLM2-Instruct (recommended)
    # alpaca = simpler Stanford instruct set
    # mix = both
    dataset_source: str = "smol-smoltalk"
    max_seq_length: int = 512
    max_train_examples: int = 10000
    val_ratio: float = 0.02
    seed: int = 1337

    # LoRA — fits T4 16GB comfortably
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    merge_lora: bool = True

    # Training
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 1000
    warmup_steps: int = 50
    logging_steps: int = 25
    save_steps: int = 250
    eval_steps: int = 250
    bf16: bool = True

    hf_repo_id: str = HF_REPO_ID
