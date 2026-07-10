"""
Instruction fine-tuning — teach the base LM to follow ### Instruction / ### Response format.

Loads checkpoints/best.pt (WikiText-2 base model) and continues training on a
small Alpaca subset.  Loss is computed only on the Response tokens; the
Instruction prefix is masked (target = -1) so the model isn't penalized for
"predicting back" the user's question.

Usage:
  python finetune.py
  python finetune.py --base checkpoints/best.pt --max-iters 200   # quick Colab test
  python finetune.py --device cuda

Output: checkpoints/chat_best.pt  (base best.pt is untouched)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import urllib.request
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    FinetuneConfig,
    ModelConfig,
    TrainConfig,
    device_summary,
    finetune_config,
    get_device,
    use_amp_on_device,
)
from instruct_format import format_instruct_prompt
from model import GPT
from tokenizer import load_tokenizer
from train import get_lr


# ---------------------------------------------------------------------------
# Alpaca download & preprocessing
# ---------------------------------------------------------------------------

def _download_url(url: str, out_path: Path) -> None:
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        if shutil.which("curl") is None:
            raise e from None
        subprocess.run(["curl", "-fsSL", url, "-o", str(out_path)], check=True)


def download_alpaca(url: str, out_path: Path) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        print(f"Downloading Alpaca dataset from {url} ...")
        _download_url(url, out_path)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"  Loaded {len(data):,} instruction examples")
    return data


def _build_instruction_text(item: dict) -> str:
    inst = item.get("instruction", "").strip()
    inp = item.get("input", "").strip()
    if inp:
        return f"{inst}\n\n{inp}"
    return inst


def prepare_instruct_examples(
    alpaca_items: list[dict],
    tokenizer,
    max_examples: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[list[int], list[int]]], list[tuple[list[int], list[int]]]]:
    """
    Tokenize Alpaca rows into (input_ids, target_ids) pairs.

    target_ids uses -1 on Instruction prefix positions (loss masking).
    """
    rng = random.Random(seed)
    items = list(alpaca_items)
    rng.shuffle(items)
    items = items[:max_examples]

    examples: list[tuple[list[int], list[int]]] = []
    skipped = 0

    for item in items:
        instruction = _build_instruction_text(item)
        response = item.get("output", "").strip()
        if not instruction or not response:
            skipped += 1
            continue

        full_text = format_instruct_prompt(instruction, response)
        prompt_only = format_instruct_prompt(instruction, None)

        full_ids = tokenizer.encode(full_text, add_special_tokens=False).ids
        prompt_len = len(tokenizer.encode(prompt_only, add_special_tokens=False).ids)

        if len(full_ids) < 4 or prompt_len >= len(full_ids):
            skipped += 1
            continue

        # Causal LM shift: x predicts next token
        x_ids = full_ids[:-1]
        y_ids = full_ids[1:]
        # Mask loss until we're inside the response (including first response token)
        for i in range(len(y_ids)):
            if i < prompt_len - 1:
                y_ids[i] = -1

        examples.append((x_ids, y_ids))

    if skipped:
        print(f"  Skipped {skipped} empty/too-short examples")

    split = int(len(examples) * (1 - val_ratio))
    train_ex = examples[:split]
    val_ex = examples[split:]
    print(f"  Train examples: {len(train_ex):,}  |  Val: {len(val_ex):,}")
    return train_ex, val_ex


# ---------------------------------------------------------------------------
# Dataset / batching
# ---------------------------------------------------------------------------

class InstructDataset(Dataset):
    def __init__(self, examples: list[tuple[list[int], list[int]]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.examples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_pad(batch: list[tuple[torch.Tensor, torch.Tensor]], max_len: int):
    """Pad variable-length sequences; pad targets with -1 (ignored by loss)."""
    xs, ys = zip(*batch)
    batch_max = min(max(x.size(0) for x in xs), max_len)
    x_pad, y_pad = [], []
    for x, y in zip(xs, ys):
        x = x[:batch_max]
        y = y[:batch_max]
        pad_len = batch_max - x.size(0)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)])
        x_pad.append(x)
        y_pad.append(y)
    return torch.stack(x_pad), torch.stack(y_pad)


@torch.no_grad()
def eval_instruct_loss(model, loader, device) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, y)
        if out.loss is not None:
            losses.append(out.loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(cfg: FinetuneConfig | None = None) -> Path:
    cfg = cfg or finetune_config
    device = get_device(cfg.device_prefer)
    amp_enabled = use_amp_on_device(device, cfg.use_amp)
    print(f"Device: {device_summary(device)}")
    if amp_enabled:
        print("Mixed precision: enabled (CUDA autocast)")

    if not cfg.base_checkpoint.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {cfg.base_checkpoint}\n"
            "Train the base model first: python train.py"
        )

    # Load base model
    print(f"\n--- Loading base checkpoint: {cfg.base_checkpoint} ---")
    state = torch.load(cfg.base_checkpoint, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**state["model_config"])
    model = GPT(mcfg)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    # Tokenizer from base run metadata
    tcfg_dict = state.get("train_config") or {}
    dataset = tcfg_dict.get("dataset_name", "wikitext2")
    from config import tokenizer_dir_for
    tokenizer = load_tokenizer(tokenizer_dir_for(dataset) / "tokenizer.json")

    # Instruction data
    print("\n--- Preparing instruction data ---")
    alpaca_path = DATA_DIR / "raw" / "alpaca_data.json"
    items = download_alpaca(cfg.alpaca_url, alpaca_path)
    train_ex, val_ex = prepare_instruct_examples(
        items, tokenizer, cfg.max_examples, cfg.val_ratio, cfg.seed,
    )

    train_loader = DataLoader(
        InstructDataset(train_ex),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate_pad(b, cfg.max_seq_len),
        num_workers=0,
    )
    val_loader = DataLoader(
        InstructDataset(val_ex),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: collate_pad(b, cfg.max_seq_len),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # Dummy TrainConfig for LR schedule reuse
    lr_cfg = TrainConfig(
        learning_rate=cfg.learning_rate,
        warmup_iters=cfg.warmup_iters,
        lr_decay_iters=cfg.lr_decay_iters,
        min_lr=cfg.min_lr,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val = float("inf")
    iter_num = 0
    train_iter = iter(train_loader)

    print(f"\n--- Fine-tuning ({cfg.max_iters} steps, lr={cfg.learning_rate}) ---")
    print(f"Output → {cfg.output_checkpoint}")

    while iter_num < cfg.max_iters:
        if iter_num % cfg.eval_interval == 0 and iter_num > 0:
            val_loss = eval_instruct_loss(model, val_loader, device)
            ppl = math.exp(min(val_loss, 20))
            print(f"step {iter_num:5d} | val loss {val_loss:.4f} | val ppl {ppl:.2f}")
            best_val = _maybe_save_instruct(
                val_loss, best_val, cfg.output_checkpoint,
                model, optimizer, iter_num, mcfg, cfg, state,
            )

        lr = get_lr(iter_num, lr_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(cfg.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            if amp_enabled:
                with torch.amp.autocast("cuda", enabled=True):
                    out = model(x, y)
                    loss = out.loss / cfg.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                out = model(x, y)
                loss = out.loss / cfg.grad_accum_steps
                loss.backward()
            running += out.loss.item()

        if amp_enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        iter_num += 1
        if iter_num % cfg.log_interval == 0:
            print(f"step {iter_num:5d} | train loss {running / cfg.grad_accum_steps:.4f} | lr {lr:.2e}")

    # Final eval + save
    val_loss = eval_instruct_loss(model, val_loader, device)
    print(f"\nFinal val loss: {val_loss:.4f}")
    prev = best_val
    best_val = _maybe_save_instruct(
        val_loss, best_val, cfg.output_checkpoint,
        model, optimizer, iter_num, mcfg, cfg, state,
    )
    if val_loss >= prev:
        print(f"  chat_best.pt unchanged (best val: {best_val:.4f})")

    print(f"\nDone. Chat checkpoint → {cfg.output_checkpoint}")
    return cfg.output_checkpoint


def _maybe_save_instruct(
    val_loss: float,
    best_val_loss: float,
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    mcfg: ModelConfig,
    fcfg: FinetuneConfig,
    base_state: dict,
) -> float:
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        _save_instruct_checkpoint(
            path, model, optimizer, iter_num, best_val_loss, mcfg, fcfg, base_state,
        )
        print(f"  ★ New best val loss: {val_loss:.4f}")
    return best_val_loss


def _save_instruct_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    best_val_loss: float,
    mcfg: ModelConfig,
    fcfg: FinetuneConfig,
    base_state: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_tcfg = base_state.get("train_config", {})
    torch.save(
        {
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "is_instruct": True,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(mcfg),
            "train_config": base_tcfg,
            "finetune_config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(fcfg).items()},
            "base_checkpoint": str(fcfg.base_checkpoint),
        },
        path,
    )
    print(f"  Instruct checkpoint saved → {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Instruction-tune a base checkpoint")
    p.add_argument("--base", type=Path, default=finetune_config.base_checkpoint)
    p.add_argument("--output", type=Path, default=finetune_config.output_checkpoint)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = p.parse_args()

    cfg = finetune_config
    cfg.base_checkpoint = args.base
    cfg.output_checkpoint = args.output
    cfg.device_prefer = args.device  # type: ignore[assignment]
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
    if args.max_examples is not None:
        cfg.max_examples = args.max_examples

    finetune(cfg)


if __name__ == "__main__":
    main()
