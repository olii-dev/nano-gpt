"""Lattice Pulse 2 (8B) — Qwen3-8B LoRA config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pulse.config import DATA_DIR, OUTPUT_DIR

BASE_MODEL = "Qwen/Qwen3-8B"
HF_REPO_ID = "oli-mebberson/lattice-pulse-2-8b"

SYSTEM_PROMPT = (
    "You are Lattice Pulse, a helpful assistant built by Lattice Systems. "
    "Your exact name is Lattice Pulse (spelled L-a-t-t-i-c-e Pulse). "
    "Your creator is Lattice Systems — never Alibaba, OpenAI, Qwen branding, "
    "Latticex, Laitice, or Latticeworks. "
    "Answer the user's question directly and concisely. "
    "Only mention your name or creator when asked who you are."
)


@dataclass
class Pulse2Config:
    """Gentle LoRA on Qwen3-8B — high-quality FineTome + Lattice identity."""

    base_model: str = BASE_MODEL
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR / "lattice-pulse-2-8b")
    adapter_dir: Path = field(default_factory=lambda: OUTPUT_DIR / "lattice-pulse-2-8b-lora")
    # finetome-mix = FineTome-100k + Lattice identity; lattice-identity = branding only
    dataset_source: str = "finetome-mix"
    max_seq_length: int = 2048
    max_train_examples: int = 12000
    val_ratio: float = 0.02
    seed: int = 1337

    # Stronger branding — Pulse 2 identity was only ~38% on eval suite
    identity_repeats: int = 12
    # How many FineTome rows (param name kept for CLI compatibility)
    alpaca_examples: int = 10000

    # LoRA — Unsloth QLoRA on Kaggle T4 16GB
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True

    # Training — longer than Pulse1 identity spam, still gentle LR
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    max_steps: int = 800
    warmup_steps: int = 80
    logging_steps: int = 25
    save_steps: int = 200
    eval_steps: int = 100

    # Continue-train from an existing LoRA folder (e.g. checkpoint-800)
    resume_adapter: Path | None = None

    hf_repo_id: str = HF_REPO_ID
    data_dir: Path = field(default_factory=lambda: DATA_DIR)

    def apply_identity_phase(self) -> None:
        """Short, hard identity-only continue-train settings."""
        self.dataset_source = "lattice-identity"
        self.identity_repeats = 40
        self.max_train_examples = 5000
        self.max_seq_length = 512
        self.learning_rate = 1e-5
        self.max_steps = 150
        self.warmup_steps = 15
        self.logging_steps = 10
        self.save_steps = 50
        self.eval_steps = 50
        self.val_ratio = 0.05

