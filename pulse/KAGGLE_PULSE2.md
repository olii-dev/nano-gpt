# Lattice Pulse 2 (8B) — Kaggle

Fine-tune **Qwen/Qwen3-8B** with Unsloth QLoRA → **Lattice Pulse 2**.

## Is FineTome-100k bad?

**No — it’s one of the best open SFT mixes.** We use it on purpose (Alpaca is out).

Watch-outs (already handled in our script):
- Soft LoRA + low LR (`2e-5`) so we don’t trash Qwen3-8B Instruct
- Light Lattice identity only (×4), not identity spam
- Qwen chat template + `enable_thinking=False`
- **Eval gate** before shipping (≥ base on smoke tests)

## Can I close my Mac?

**Only if you use Save Version → Save & Run All (Commit).**

| How you run | Close Mac? | What happens |
|-------------|------------|--------------|
| Click ▶ on cells in the editor | **No** | Session dies when the tab/browser disconnects |
| **Save Version → Save & Run All** | **Yes** | Independent cloud job; come back later for logs + Output |

Session limit is still ~9–12 hours of GPU. Full 800 steps may need a second version later — download the adapter from Output either way.

---

## What to do (background — recommended)

### 1. New notebook (you said you did this)
- Accelerator: **GPU T4 x2** (or T4)
- Internet: **On**

### 2. Put everything in cells (don’t rely on interactive ▶ only)

**Cell 1**
```python
!pip install -q unsloth
!pip uninstall -y torchao 2>/dev/null; true
```

**Cell 2**
```python
!rm -rf /kaggle/working/nano-gpt
!git clone https://github.com/olii-dev/nano-gpt.git /kaggle/working/nano-gpt
%cd /kaggle/working/nano-gpt
!git log -1 --oneline
```
You should see a commit message about Pulse 2 / Unsloth.

**Cell 3 — train**
```python
!python -m pulse.train_unsloth --device cuda
```

Optional smoke first (interactive only, then switch to full for Save Version):
```python
# !python -m pulse.train_unsloth --device cuda --max-steps 50
```

### 3. Start the background job
1. Top right → **Save Version**
2. Choose **Save & Run All (Commit)**
3. Confirm **Save**
4. You can close the Mac

### 4. Come back later
1. Open the notebook → **Versions** (left / top)
2. Open the finished version
3. Check **Logs** for errors / “Done. Adapter…”
4. **Output** → download `lattice-pulse-2-8b-lora/`

### 5. Before calling it a win
Smoke-test identity + a few general Qs; compare feel vs base `Qwen/Qwen3-8B`.  
Only keep if identity is Lattice **and** it doesn’t feel dumber than base.

Copy adapter to Proton: `Lattice Models/Pulse2/`.

---

## Hyperparams we use (vs generic AI advice)

| Setting | Ours | Why |
|---------|------|-----|
| Dataset | FineTome-100k + light identity | Quality > Alpaca |
| Quant | 4-bit QLoRA | Fits T4 |
| LoRA r / α | 16 / 32 | Stable |
| LR | **2e-5** | Gentler than 2e-4 on an Instruct 8B |
| Max steps | 800 | ~1 pass over mix, not multi-epoch spam |
| Framework | Unsloth | Faster / less VRAM on Kaggle |

Unsloth will typically use **one** T4; that’s OK — don’t fight for multi-GPU unless you know what you’re doing.

## License

Qwen3 is Apache-2.0 — fine-tune + sell access OK; attribute the base on the model card.
