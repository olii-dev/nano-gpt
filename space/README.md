---
title: Lattice Mini
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: mit
---

# Lattice Mini

**Lattice Mini** is a compact language model from **Lattice Systems** — built **entirely from scratch** at ~42M parameters.

| Spec | Value |
|------|-------|
| Architecture | 12-layer GPT decoder |
| Context | 512 tokens |
| Vocab | ~8k BPE |
| Pretraining | WikiText-2 |
| Alignment | Alpaca instruction tuning |

**This is a research demo.** Answers are often wrong or repetitive — that's expected at this scale.

## Try asking

- "What is the capital of France?"
- "Write a short poem about the ocean."
- "Explain gravity in simple terms."

Lower **temperature** (0.3–0.5) usually helps.

## How it was built

1. Custom BPE tokenizer
2. Pretrain on WikiText-2
3. Instruction-tune on Alpaca (chat format)
4. Serve via Gradio on CPU

## Source code

Training pipeline: [github.com/olii-dev/nano-gpt](https://github.com/olii-dev/nano-gpt)
