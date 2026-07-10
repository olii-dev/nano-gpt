"""
Training loop for the small GPT model.

Features:
  - Device priority: CUDA (Colab T4) > MPS (Apple Silicon) > CPU
  - CUDA: optional mixed precision (autocast + GradScaler) for faster training
  - MPS/CPU: float32 only (MPS mixed precision is unstable)
  - Cosine LR schedule with linear warmup
  - Gradient accumulation for effective larger batch sizes
  - Periodic validation loss + perplexity
  - CSV metrics log + optional loss plot
  - Checkpointing with resume support
  - Graceful Ctrl+C → saves checkpoint before exit

Usage:
  python train.py                     # fresh run
  python train.py --resume latest     # resume from newest checkpoint
  python train.py --resume checkpoints/ckpt_1500.pt
  python train.py --max-iters 50      # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    ModelConfig,
    TrainConfig,
    count_parameters,
    device_summary,
    get_device,
    model_config,
    save_config,
    train_config,
    use_amp_on_device,
)
from dataset import get_batch, prepare_dataset
from model import GPT, build_model


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    """
    Linear warmup → cosine decay to min_lr.

    Warmup prevents early instability when gradients are large.
    Cosine decay gently anneals the learning rate so the model settles
    into a good minimum rather than oscillating forever.
    """
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / max(cfg.warmup_iters, 1)

    if step > cfg.lr_decay_iters:
        return cfg.min_lr

    decay_ratio = (step - cfg.warmup_iters) / max(cfg.lr_decay_iters - cfg.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model: GPT,
    prepared,
    eval_iters: int,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Average loss on random train/val batches (quick proxy for full eval)."""
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = []
        for i in range(eval_iters):
            x, y = get_batch(split, prepared, block_size, batch_size, device, seed=i)
            logits = model(x, y)
            losses.append(logits.loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    best_val_loss: float,
    mcfg: ModelConfig,
    tcfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Always save on CPU for portability across devices
    torch.save(
        {
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(mcfg),
            "train_config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(tcfg).items()},
        },
        path,
    )
    print(f"  Checkpoint saved → {path}")


def maybe_save_best(
    val_loss: float,
    best_val_loss: float,
    checkpoint_path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    mcfg: ModelConfig,
    tcfg: TrainConfig,
) -> float:
    """
    Save best.pt only when val loss improves.

    Returns the updated best_val_loss. Periodic and final eval both use this
    so training can run to max_iters without overwriting the best checkpoint.
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            checkpoint_path, model, optimizer, iter_num, best_val_loss, mcfg, tcfg,
        )
        print(f"  ★ New best val loss: {val_loss:.4f}")
    return best_val_loss


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    ckpts = sorted(checkpoint_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def load_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    map_location = device or "cpu"
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Append-only CSV log + optional matplotlib plot at end."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / "train_metrics.csv"
        self.rows: list[dict] = []
        self._header_written = self.csv_path.exists()

    def log(self, row: dict) -> None:
        self.rows.append(row)
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def plot(self) -> Path | None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        if not self.rows:
            return None

        steps = [r["step"] for r in self.rows]
        train_loss = [r["train_loss"] for r in self.rows if r.get("train_loss") is not None]
        val_loss = [r["val_loss"] for r in self.rows if r.get("val_loss") is not None]

        fig, ax = plt.subplots(figsize=(8, 4))
        if train_loss:
            ax.plot(steps[: len(train_loss)], train_loss, label="train loss")
        val_steps = [r["step"] for r in self.rows if r.get("val_loss") is not None]
        if val_loss:
            ax.plot(val_steps, val_loss, label="val loss", marker="o")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
        ax.set_title("Training progress")
        fig.tight_layout()
        out = self.csv_path.parent / "loss_curve.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    mcfg: ModelConfig | None = None,
    tcfg: TrainConfig | None = None,
    resume_path: Path | None = None,
    max_iters_override: int | None = None,
) -> Path:
    mcfg = mcfg or model_config
    tcfg = tcfg or train_config
    if max_iters_override is not None:
        tcfg.max_iters = max_iters_override

    # Reproducibility
    torch.manual_seed(tcfg.seed)

    device = get_device(tcfg.device_prefer)
    amp_enabled = use_amp_on_device(device, tcfg.use_amp)
    print(f"Device: {device_summary(device)}")
    if amp_enabled:
        print("Mixed precision: enabled (CUDA autocast)")
    elif device.type == "mps":
        print("Mixed precision: disabled (MPS — float32 only)")
    else:
        print("Mixed precision: disabled")

    # Data
    print("\n--- Preparing dataset ---")
    prepared = prepare_dataset(
        name=tcfg.dataset_name,
        train_ratio=tcfg.train_split_ratio,
        vocab_size=mcfg.vocab_size,
        seed=tcfg.seed,
    )

    # Sync vocab size with actual tokenizer output
    if prepared.vocab_size != mcfg.vocab_size:
        print(f"Adjusting model vocab_size: {mcfg.vocab_size} → {prepared.vocab_size}")
        mcfg.vocab_size = prepared.vocab_size

    # Model
    print("\n--- Building model ---")
    model = build_model(mcfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.learning_rate,
        betas=(tcfg.beta1, tcfg.beta2),
        weight_decay=tcfg.weight_decay,
    )

    iter_num = 0
    best_val_loss = float("inf")

    if resume_path is not None:
        if resume_path == Path("latest"):
            resume_path = find_latest_checkpoint(tcfg.checkpoint_dir)
        if resume_path and resume_path.exists():
            print(f"\nResuming from {resume_path}")
            meta = load_checkpoint(resume_path, model, optimizer, device)
            iter_num = meta.get("iter_num", 0)
            best_val_loss = meta.get("best_val_loss", float("inf"))
        else:
            print("No checkpoint found — starting fresh.")

    # Save run config for reproducibility
    save_config(tcfg.checkpoint_dir / "run_config.json", model=mcfg, train=tcfg)

    logger = MetricsLogger(tcfg.log_dir)
    tokens_per_iter = tcfg.batch_size * mcfg.block_size * tcfg.grad_accum_steps

    # Ctrl+C handler — save before exit
    interrupted = False

    def _handle_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\nInterrupt received — saving checkpoint before exit ...")

    signal.signal(signal.SIGINT, _handle_sigint)

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"\n--- Training ({tcfg.max_iters} steps, effective batch {tcfg.batch_size * tcfg.grad_accum_steps}) ---")
    print(f"Parameters: {count_parameters(model):,}")
    t0 = time.time()
    running_loss = 0.0

    while iter_num < tcfg.max_iters and not interrupted:
        # Periodic validation
        if iter_num % tcfg.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(
                model, prepared, tcfg.eval_iters, mcfg.block_size, tcfg.batch_size, device
            )
            val_ppl = math.exp(min(losses["val"], 20))  # cap to avoid overflow
            print(
                f"step {iter_num:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | val ppl {val_ppl:.2f}"
            )
            logger.log({
                "step": iter_num,
                "train_loss": None,
                "val_loss": losses["val"],
                "val_ppl": val_ppl,
                "lr": get_lr(iter_num, tcfg),
                "tokens_seen": iter_num * tokens_per_iter,
            })
            best_val_loss = maybe_save_best(
                losses["val"], best_val_loss,
                tcfg.checkpoint_dir / "best.pt",
                model, optimizer, iter_num, mcfg, tcfg,
            )

        # LR for this step
        lr = get_lr(iter_num, tcfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation micro-steps
        optimizer.zero_grad(set_to_none=True)
        for micro in range(tcfg.grad_accum_steps):
            x, y = get_batch(
                "train", prepared, mcfg.block_size, tcfg.batch_size, device,
                seed=iter_num * tcfg.grad_accum_steps + micro,
            )
            if amp_enabled:
                with torch.amp.autocast("cuda", enabled=True):
                    out = model(x, y)
                    loss = out.loss / tcfg.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                out = model(x, y)
                loss = out.loss / tcfg.grad_accum_steps
                loss.backward()
            running_loss += out.loss.item()

        if amp_enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optimizer.step()
        iter_num += 1

        if iter_num % tcfg.log_interval == 0:
            avg_loss = running_loss / (tcfg.log_interval * tcfg.grad_accum_steps)
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            print(
                f"step {iter_num:5d} | loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                f"lr {lr:.2e} | {elapsed:.0f}s"
            )
            logger.log({
                "step": iter_num,
                "train_loss": avg_loss,
                "val_loss": None,
                "val_ppl": None,
                "lr": lr,
                "tokens_seen": iter_num * tokens_per_iter,
            })
            running_loss = 0.0

        if iter_num % tcfg.save_interval == 0:
            save_checkpoint(
                tcfg.checkpoint_dir / f"ckpt_{iter_num}.pt",
                model, optimizer, iter_num, best_val_loss, mcfg, tcfg,
            )

    # Final save + eval
    if iter_num > 0:
        losses = estimate_loss(
            model, prepared, tcfg.eval_iters, mcfg.block_size, tcfg.batch_size, device
        )
        val_ppl = math.exp(min(losses["val"], 20))
        print(
            f"\nFinal eval | train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f} | val ppl {val_ppl:.2f}"
        )
        prev_best = best_val_loss
        best_val_loss = maybe_save_best(
            losses["val"], best_val_loss,
            tcfg.checkpoint_dir / "best.pt",
            model, optimizer, iter_num, mcfg, tcfg,
        )
        if losses["val"] >= prev_best:
            print(f"  best.pt unchanged (best val loss: {best_val_loss:.4f} from earlier step)")

    final_path = tcfg.checkpoint_dir / f"ckpt_{iter_num}.pt"
    save_checkpoint(final_path, model, optimizer, iter_num, best_val_loss, mcfg, tcfg)

    elapsed = time.time() - t0
    plot_path = logger.plot()
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Final step: {iter_num}")
    if plot_path:
        print(f"Loss plot saved → {plot_path}")
    print(f"Best checkpoint → {tcfg.checkpoint_dir / 'best.pt'}")

    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the small GPT model")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint, or 'latest'")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--dataset", choices=["wikitext2", "tiny_shakespeare"], default=None,
                        help="Training corpus (default: config train_config.dataset_name)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA mixed precision even on GPU")
    args = parser.parse_args()

    tcfg = train_config
    if args.dataset is not None:
        tcfg.dataset_name = args.dataset
    tcfg.device_prefer = args.device  # type: ignore[assignment]
    if args.no_amp:
        tcfg.use_amp = False

    resume = Path(args.resume) if args.resume else None
    if args.resume == "latest":
        resume = Path("latest")

    train(resume_path=resume, max_iters_override=args.max_iters)


if __name__ == "__main__":
    main()
