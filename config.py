"""
Central configuration for the small language model project.

All hyperparameters live here so you can experiment in one place without
hunting through train.py / model.py.  Device selection priority:
  CUDA (Colab / NVIDIA GPU) > MPS (Apple Silicon) > CPU
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

DevicePrefer = Literal["auto", "cuda", "mps", "cpu"]

# ---------------------------------------------------------------------------
# Device helpers — CUDA > MPS > CPU
# ---------------------------------------------------------------------------
#
# Device-specific behavior elsewhere in the project:
#
#   CUDA (Colab T4, etc.):
#     - train.py: optional mixed precision (autocast + GradScaler) via use_amp
#     - model.py: fused scaled_dot_product_attention (faster than manual matmul)
#     - model.py: sampling (multinomial, top-k/top-p) stays on GPU
#
#   MPS (Apple Silicon):
#     - train.py: float32 only — MPS mixed precision is unstable
#     - model.py: manual attention matmul (SDPA / flash attn flaky on MPS)
#     - model.py: sampling moved to CPU (torch.multinomial MPS bugs)
#
#   CPU:
#     - Same attention path as MPS; no AMP
# ---------------------------------------------------------------------------

def _mps_usable() -> bool:
    """True if MPS is reported available and passes a smoke test."""
    if not torch.backends.mps.is_available():
        return False
    try:
        t = torch.zeros(1, device="mps")
        _ = t + 1
        return True
    except Exception:
        return False


def get_device(prefer: DevicePrefer = "auto") -> torch.device:
    """
    Pick the best available compute device.

    Priority when prefer="auto": CUDA > MPS > CPU.
    Explicit prefer="cuda" / "mps" / "cpu" forces that backend (with fallback
    to CPU only when the requested backend is unavailable).
    """
    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Warning: CUDA requested but unavailable — falling back to CPU")
        return torch.device("cpu")

    if prefer == "mps":
        if _mps_usable():
            return torch.device("mps")
        print("Warning: MPS requested but unavailable — falling back to CPU")
        return torch.device("cpu")

    # auto — CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_usable():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary(device: torch.device) -> str:
    """Human-readable device info for logging."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"cuda ({name})"
    if device.type == "mps":
        return "mps (Apple Metal — accelerated)"
    return "cpu (no GPU backend available)"


def device_needs_mps_workarounds(device: torch.device) -> bool:
    """True when model.py should use manual attention + CPU-side sampling."""
    return device.type == "mps"


def use_amp_on_device(device: torch.device, enabled: bool = True) -> bool:
    """Mixed precision training is CUDA-only; disabled on MPS/CPU."""
    return enabled and device.type == "cuda"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
LOG_DIR = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, CHECKPOINT_DIR, TOKENIZER_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model & training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    GPT-style decoder-only transformer.

  Target: ~25–50M parameters.  Current defaults land around ~42M:
      vocab=8000, n_layer=12, n_embd=512, n_head=8, block_size=512

  Scaling guide (32 GB unified memory, MPS):
    - More layers / wider embeddings: linear-ish VRAM growth (~4 * n_embd^2
      per block for attention + FFN weights).
    - Longer context (block_size): grows position embeddings and attention
      activations as O(block_size^2) per layer during training.
    - Rough ceiling on M4 32GB with float32: ~200–350M params at 512 context
      if batch_size=1–4; beyond that you'll need gradient checkpointing,
      shorter context, or CPU offloading.  1B+ is impractical without
      quantization and/or multi-machine training.
    """

    vocab_size: int = 8000
    n_layer: int = 12
    n_embd: int = 512
    n_head: int = 8
    block_size: int = 512          # max context length (tokens)
    dropout: float = 0.1
    bias: bool = False             # False matches modern GPT-2/3 style

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    """Training loop settings."""

    # Data
    dataset_name: str = "tiny_shakespeare"   # see dataset.py for options
    train_split_ratio: float = 0.9
    val_split_ratio: float = 0.1

    # Optimization
    # use_amp: CUDA-only mixed precision (autocast). Forced off on MPS/CPU.
    batch_size: int = 8
    grad_accum_steps: int = 4        # effective batch = 32
    use_amp: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_iters: int = 3000            # enough for ~1 MB Tiny Shakespeare

    # Schedule (cosine with warmup)
    warmup_iters: int = 100
    lr_decay_iters: int = 3000
    min_lr: float = 3e-5

    # Logging & checkpointing
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    save_interval: int = 500

    # Reproducibility
    seed: int = 1337

    # Device preference passed to get_device()
    device_prefer: DevicePrefer = "auto"

    # Paths (filled relative to project root)
    checkpoint_dir: Path = field(default_factory=lambda: CHECKPOINT_DIR)
    log_dir: Path = field(default_factory=lambda: LOG_DIR)


@dataclass
class GenerateConfig:
    """Inference / sampling settings."""

    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float | None = 0.9
    seed: int | None = None


# Default singletons — import these or build your own
model_config = ModelConfig()
train_config = TrainConfig()
generate_config = GenerateConfig()


def save_config(path: Path, **configs) -> None:
    """Serialize configs to JSON (Path values become strings)."""
    payload = {}
    for name, cfg in configs.items():
        d = asdict(cfg)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        payload[name] = d
    path.write_text(json.dumps(payload, indent=2))


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
