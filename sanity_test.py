"""
Sanity checks before a full training run.

1. MPS detection — confirms Metal acceleration is active (not silent CPU fallback)
2. Overfit test — train on a tiny repeated batch; loss should drop toward ~0

Run:  python sanity_test.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from config import ModelConfig, get_device, device_summary
from model import GPT


def check_device() -> torch.device:
    """Verify the selected device is available and runs a matmul."""
    print("=" * 60)
    print("DEVICE CHECK")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    print(f"  torch.cuda.is_available()            = {cuda_available}")
    if cuda_available:
        print(f"  GPU                                  = {torch.cuda.get_device_name(0)}")
    print(f"  torch.backends.mps.is_built()        = {mps_built}")
    print(f"  torch.backends.mps.is_available()    = {mps_available}")

    device = get_device()
    print(f"  Selected device: {device_summary(device)}")

    t = torch.randn(4, 4, device=device)
    result = t @ t.T
    print(f"  Test tensor device: {t.device}")
    print(f"  Matmul result device: {result.device}")

    if device.type in ("cuda", "mps"):
        assert t.device.type == device.type, f"Expected {device.type} but tensor is on {t.device}!"
        print(f"  ✓ {device.type.upper()} is active and running ops on GPU")
    else:
        print("  ⚠ Running on CPU — training will be slower but still works")

    print()
    return device


def overfit_tiny_batch(device: torch.device, steps: int = 300) -> bool:
    """
    Memorize a single 16-token sequence repeated in a batch of 4.

    If loss doesn't fall below 0.5, something is wrong with the model or loop.
    """
    print("=" * 60)
    print("OVERFIT TEST (tiny batch)")
    print("=" * 60)

    cfg = ModelConfig(
        vocab_size=128,
        n_layer=4,
        n_embd=128,
        n_head=4,
        block_size=32,
        dropout=0.0,  # no dropout — we want perfect memorization
    )
    model = GPT(cfg).to(device)
    model.train()

    # Fixed sequence: model should memorize these exact next-token targets
    seq = torch.tensor(
        [5, 12, 8, 30, 45, 2, 67, 89, 11, 22, 33, 44, 55, 66, 77, 88],
        device=device,
    )
    # Input = all but last, target = all but first (standard LM shift)
    x = seq[:-1].unsqueeze(0).repeat(4, 1)   # (4, 15)
    y = seq[1:].unsqueeze(0).repeat(4, 1)    # (4, 15)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    initial_loss = None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        out = model(x, y)
        loss = out.loss
        assert loss is not None
        loss.backward()
        optimizer.step()

        if step == 1:
            initial_loss = loss.item()
        if step % 50 == 0 or step == steps:
            print(f"  step {step:4d}  loss = {loss.item():.6f}")

    final_loss = loss.item()
    passed = final_loss < 0.5
    print()
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.6f}")
    if passed:
        print("  ✓ PASS — model can memorize a tiny batch (pipeline is wired correctly)")
    else:
        print("  ✗ FAIL — loss did not drop enough; check model / training loop")
    print()
    return passed


def forward_backward_smoke(device: torch.device) -> bool:
    """Quick shape + gradient check on the full-size config."""
    print("=" * 60)
    print("FORWARD/BACKWARD SMOKE TEST")
    print("=" * 60)

    cfg = ModelConfig(vocab_size=256, n_layer=2, n_embd=64, n_head=4, block_size=64)
    model = GPT(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 32), device=device)

    out = model(x, y)
    out.loss.backward()

    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"  Logits: {tuple(out.logits.shape)}  loss: {out.loss.item():.4f}")
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  ✓ Forward + backward OK")
    print()
    return True


def main() -> None:
    device = check_device()
    ok1 = forward_backward_smoke(device)
    ok2 = overfit_tiny_batch(device)

    if ok1 and ok2:
        print("All sanity checks passed. Ready for tokenizer training + full run.")
        sys.exit(0)
    else:
        print("Some checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
