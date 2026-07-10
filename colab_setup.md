# Google Colab setup

Train the 42M-parameter GPT on a free T4 GPU. Colab wipes all files when the
runtime disconnects — **download `best.pt` before you close the tab**.

## Before you start

1. Create a GitHub repo and push this project (see README / main docs).
2. In Colab: **Runtime → Change runtime type → T4 GPU → Save**.

## Notebook cells

Copy each block into its own Colab cell and run top-to-bottom.

---

### Cell 1 — Verify GPU

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("No GPU! Go to Runtime → Change runtime type → T4 GPU")
```

---

### Cell 2 — Clone repo

Replace `YOUR_USERNAME` and `REPO_NAME` with your GitHub details.

```python
!rm -rf REPO_NAME
!git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
%cd REPO_NAME
```

---

### Cell 3 — Install dependencies

Colab ships with PyTorch + CUDA; we only need the rest.

```python
!pip install -q tokenizers tqdm matplotlib
```

---

### Cell 4 — Sanity check (optional, ~1 min)

```python
!python sanity_test.py
```

You should see `cuda (Tesla T4)` (or similar) as the selected device.

---

### Cell 5 — Download data + train tokenizer

```python
!python dataset.py
```

---

### Cell 6 — Train (~1–2 hours on T4 with AMP)

```python
!python train.py --device cuda
```

`--device cuda` is optional on Colab (`auto` picks CUDA anyway). Mixed
precision is enabled automatically on CUDA.

**Resume** if the runtime dies mid-training:

```python
!python train.py --device cuda --resume latest
```

---

### Cell 7 — Quick generation test

```python
!python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:" --max-new-tokens 120
```

---

### Cell 8 — Download checkpoint (do this before disconnecting!)

```python
from google.colab import files
files.download("checkpoints/best.pt")
```

Optional: download the tokenizer and loss plot too.

```python
files.download("tokenizer/tokenizer.json")
files.download("logs/loss_curve.png")
```

---

## Using the checkpoint locally

Copy `best.pt` into your local `checkpoints/` folder (alongside the trained
`tokenizer/tokenizer.json` from the same run), then:

```bash
source .venv/bin/activate
python generate.py --checkpoint checkpoints/best.pt --chat
```

## Colab vs local M4 (rough)

| | Colab T4 (CUDA + AMP) | M4 MPS (float32) |
|--|----------------------|------------------|
| ~3000 steps | ~1–2 hours | ~4–6 hours |
| Attention | Fused SDPA | Manual matmul |
| Precision | fp16 autocast | float32 only |

## Troubleshooting

- **Out of memory** — in `config.py` reduce `batch_size` to `4` or `block_size` to `256`.
- **Session expired** — re-run cells 2–6 with `--resume latest`.
- **Slow first step** — CUDA kernel warmup; speed stabilizes after ~10 steps.
