# Lattice Pulse 2 (8B) — Kaggle

Fine-tune **Qwen/Qwen3-8B** with Unsloth QLoRA → **Lattice Pulse 2**.

## FineTome-100k?

**Good choice** — curated high-quality SFT (not Alpaca). We already use it.

## Background run (close your Mac) — required path

This matches Kaggle’s own docs ([Notebooks](https://www.kaggle.com/docs/Notebooks)) and the usual “run kernel in background” advice ([discussion](https://www.kaggle.com/discussions/general/66860)):

> **Save & Run All** creates a **new session** with a clean state and runs the notebook **top to bottom**. That session is **separate** from the interactive editor — you can close the browser / Mac.

| Method | Close Mac? | Notes |
|--------|------------|--------|
| Click ▶ / Run All in the editor | **No** | Interactive session; idle timeout kills it |
| **Save Version → Quick-Save** | No training | Snapshot only — does **not** run cells |
| **Save Version → Save & Run All (Commit)** | **Yes** | Real background job; outputs land under Versions |

Hard limits:
- Entire Save & Run All must finish within **~12 hours** (GPU)
- Interactive editing idle timeout is short (~20 min) — irrelevant if you use Save & Run All

---

## Checklist before Save & Run All

Do these in the **editor** first (they apply to the version):

1. **Settings → Accelerator → GPU T4 x2** (or T4)
2. **Settings → Internet → On**
3. Notebook cells are complete top→bottom (no “run this cell later by hand”)
4. No secrets you forgot to add (we don’t need HF token for train-only)

Then put **exactly** these cells:

### Cell 1
```python
!pip install -q unsloth
!pip uninstall -y torchao 2>/dev/null; true
```

### Cell 2
```python
!rm -rf /kaggle/working/nano-gpt
!git clone https://github.com/olii-dev/nano-gpt.git /kaggle/working/nano-gpt
%cd /kaggle/working/nano-gpt
!git log -1 --oneline
```

### Cell 3
```python
!python -m pulse.train_unsloth --device cuda
```

### Start background job
1. Top right → **Save Version**
2. Select **Save & Run All** (aka Commit) — **not** Quick-Save
3. Optional: name it `pulse2-full`
4. Click **Save**
5. Wait until the version shows as **running** (Versions panel / viewer)
6. **Now** you can close the Mac

### Come back
1. Open the notebook → **Versions** (or “View versions”)
2. Open the finished version
3. **Logs** — look for errors or “Done. Adapter…”
4. **Output** — download `lattice-pulse-2-8b-lora/`

If the version **fails**, fix the cells, Save & Run All again.  
If it **times out** at 12h, lower steps next run, e.g.:
```python
!python -m pulse.train_unsloth --device cuda --max-steps 400
```

---

## Optional: interactive smoke (keep tab open)

Only to verify install/clone before the real job:

```python
!python -m pulse.train_unsloth --device cuda --max-steps 20
```

Then change Cell 3 back to full train and do **Save & Run All**.

---

## After download

1. **Keep only checkpoint-400** (best eval) — delete other checkpoints (~800MB saved).
2. Smoke-test: `python -m pulse.chat_pulse2 --adapter path/to/checkpoint-400`
3. Benchmark: `python -m pulse.benchmark --compare pulse2,qwen3 --device cuda`
4. Copy to Proton: `Lattice Models/Pulse2/`

### Retrain (Pulse 2.1 — stronger identity)

- Identity data: `pulse/data/lattice_custom.json` (~80 examples)
- Repeated **12×** in FineTome mix (`identity_repeats` in `config_pulse2.py`)
- Push latest `nano-gpt` to GitHub, then Kaggle: `!python -m pulse.train_unsloth --device cuda`
- After train, pick checkpoint with lowest eval loss (often ~step 400)

## Hyperparams (ours)

| Setting | Value |
|---------|--------|
| Data | FineTome-100k + Lattice identity (12× repeat) |
| Quant | 4-bit QLoRA |
| LoRA r / α | 16 / 32 |
| LR | 2e-5 |
| Steps | 800 (cut if near 12h limit) |
| Framework | Unsloth |

## License

Qwen3 Apache-2.0 — fine-tune + sell access OK; attribute the base.
