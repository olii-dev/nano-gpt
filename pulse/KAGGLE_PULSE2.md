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

1. Smoke-test checkpoints (prefer **800** from chat samples; compare with 400).
2. Benchmark winner: `python -m pulse.benchmark --compare pulse2,qwen3 --device mps`
3. Copy winner to Proton: `Lattice Models/Pulse2/`

### Identity continue-train (Pulse 2.2) — do this when branding still flops

Full FineTome mixes leave ~9% identity data — not enough to beat Qwen3’s Alibaba prior.
Run a **short identity-only** continue from **checkpoint-800** (~30–60 min on T4).

**A. Upload adapter to Kaggle as a dataset**
1. Zip only the folder: `checkpoint-800/` (must contain `adapter_model.safetensors` + `adapter_config.json`)
2. Kaggle → **Datasets** → New Dataset → upload zip → name it e.g. `pulse2-ckpt800`
3. In your notebook: **Add Input** → that dataset

**B. Notebook cells**
```python
# Cell 1
!pip install -q unsloth
!pip uninstall -y torchao 2>/dev/null; true
```
```python
# Cell 2
!rm -rf /kaggle/working/nano-gpt
!git clone https://github.com/olii-dev/nano-gpt.git /kaggle/working/nano-gpt
%cd /kaggle/working/nano-gpt
!git log -1 --oneline
```
```python
# Cell 3 — find your uploaded checkpoint path, then train
import os
from pathlib import Path
cands = list(Path("/kaggle/input").rglob("adapter_model.safetensors"))
print("found:", cands)
assert cands, "Add the pulse2-ckpt800 dataset as Input"
resume = str(cands[0].parent)
print("resume:", resume)
!python -m pulse.train_unsloth --device cuda --identity-only --resume-adapter {resume}
```

**C. Save & Run All**, download `lattice-pulse-2-8b-identity/`, chat-test with `--greedy`.

Recipe: identity JSON ×40, 150 steps, LR `1e-5`, seq 512. Not a guarantee — but the right fix for Alibaba/spelling bleed.

## Hyperparams (ours)

| Setting | Full mix | Identity phase |
|---------|----------|----------------|
| Data | FineTome + identity 12× | identity only 40× |
| Steps | 800 | 150 |
| LR | 2e-5 | 1e-5 |

## License

Qwen3 Apache-2.0 — fine-tune + sell access OK; attribute the base.
