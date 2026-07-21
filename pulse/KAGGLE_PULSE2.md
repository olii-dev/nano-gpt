# Lattice Pulse 2 (8B) — Kaggle

Fine-tune **Qwen/Qwen3-8B** with Unsloth QLoRA → **Lattice Pulse 2**.

## Training data (not Alpaca)

| Source | Role |
|--------|------|
| **[mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)** | High-quality multi-turn chat, reasoning, tools — Unsloth’s recommended SFT mix |
| **Clean Lattice identity** (`lattice_custom.json` ×4) | Brand only — light so we don’t overwrite Qwen skills |

Alpaca is **out** — too weak to beat Qwen3-8B.

### Reality check

Beating the base **on every benchmark** with a free-tier LoRA is hard (Qwen3-8B is already heavily post-trained). This mix is the best honest free attempt:

- Aim: **≥ base** on HellaSwag / chat smoke + **much better** identity  
- Ship only if eval gate passes — otherwise tune data/steps, don’t publish a worse model

## One-time Kaggle setup

1. **Create → New Notebook**
2. **Settings → Accelerator → GPU T4 x2** (or T4)
3. **Internet → On**
4. Paste the cells below

## Cell 1 — Install

```python
!pip install -q unsloth
!pip uninstall -y torchao 2>/dev/null; true
```

## Cell 2 — Clone repo

```python
!rm -rf /kaggle/working/nano-gpt
!git clone https://github.com/olii-dev/nano-gpt.git /kaggle/working/nano-gpt
%cd /kaggle/working/nano-gpt
```

Push the latest Pulse 2 files to GitHub first, or upload `pulse/data/lattice_custom.json` + `pulse/train_unsloth.py` into Inputs.

## Cell 3 — Train

Full run (~several hours on T4 — may need resume / multi-session):

```python
!python -m pulse.train_unsloth --device cuda
```

Smoke (50 steps):

```python
!python -m pulse.train_unsloth --device cuda --max-steps 50
```

## Cell 4 — Compare vs base (must pass before shipping)

```python
from unsloth import FastLanguageModel

def ask(model, tokenizer, q):
    messages = [
        {"role": "system", "content": "You are Lattice Pulse, a helpful assistant built by Lattice Systems. Answer the user's question directly and concisely. Only mention your name or creator when asked who you are."},
        {"role": "user", "content": q},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=100, temperature=0.2, do_sample=True)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

# Pulse 2 adapter
p2, tok = FastLanguageModel.from_pretrained(
    "/kaggle/working/lattice-pulse-2-8b-lora", max_seq_length=2048, load_in_4bit=True
)
FastLanguageModel.for_inference(p2)

for q in ["Who are you?", "Explain gravity in one sentence.", "What is 17+25?"]:
    print("Q:", q)
    print("A:", ask(p2, tok, q))
    print("---")
```

Also run your HellaSwag smoke vs `Qwen/Qwen3-8B` before calling it a win.

## After training

1. Download `lattice-pulse-2-8b-lora/`
2. Copy to Proton: `Lattice Models/Pulse2/`
3. Push HF only if eval ≥ base on the gates above

## License

Qwen3 is Apache-2.0 — fine-tune + sell access OK; attribute the base on the model card.
