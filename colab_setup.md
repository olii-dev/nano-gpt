# Google Colab setup

Train the ~148M-parameter GPT (Lattice Standard) on a free T4 GPU. Colab wipes all
files when the runtime disconnects — **download `best.pt` before you close the tab**.

Checkpoints are ~1.8 GB each at this size (vs ~510 MB for the old 42M model).

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

WikiText-2 is the default. For Tiny Shakespeare, add `--dataset tiny_shakespeare`.

```python
!python dataset.py --dataset wikitext2
```

---

### Cell 6 — Train (~2–3 hours on T4 with AMP, 10k steps)

```python
!python train.py --device cuda --dataset wikitext2
```

`--device cuda` is optional on Colab (`auto` picks CUDA anyway). Mixed
precision is enabled automatically on CUDA.

**Resume** if the runtime dies mid-training:

```python
!python train.py --device cuda --resume latest
```

---

### Cell 7 — Instruction fine-tune for chat (~15–30 min on T4)

Teaches the base model to answer in `### Instruction / ### Response` format.

```python
!python finetune.py --device cuda --base checkpoints/best.pt
```

Quick smoke test (~5 min):

```python
!python finetune.py --device cuda --max-iters 100 --max-examples 1000
```

---

### Cell 8 — Test chat model

```python
!python generate.py --checkpoint checkpoints/chat_best.pt --chat
```

Base model (Wikipedia continuation) still works with `--base`:

```python
!python generate.py -c checkpoints/best.pt --base --prompt "The city is located in"
```

---

### Cell 9 — Download checkpoints (before disconnecting!)

```python
from google.colab import files
files.download("checkpoints/chat_best.pt")
files.download("checkpoints/best.pt")
files.download("tokenizer/wikitext2/tokenizer.json")
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
