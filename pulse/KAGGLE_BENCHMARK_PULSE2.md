# Pulse 2 vs Base Qwen3-8B — Kaggle Benchmark

Head-to-head: did the QLoRA fine-tune actually change anything?
Runs each model on the **same** identity + factual suites, under **two** system-prompt
conditions so you can separate "the prompt did it" from "the fine-tune did it".

## What it measures

| Suite | What | Pass criteria |
|-------|------|---------------|
| **Identity** (8 prompts) | "who are you / who made you / spell your name" | says "Lattice", no bad brands (Alibaba/Qwen/ChatGPT/Latticex/...) |
| **Factual** (8 prompts) | capitals, math, trivia | correct answer token present |

Two **modes** (the honest bit):
- `with_system` — uses the *same system prompt the live server sends*. This is what visitors experience.
- `no_system` — neutral prompt only. **This isolates what the fine-tune actually contributes** vs what the system prompt is doing. (Expect both models to fail identity here — that tells you the fine-tune is weak.)

## Run it

Kaggle notebook: **GPU T4 x2** (or T4), **Internet ON**.

### Cell 1 — deps
```python
!pip install -q "transformers>=4.51" "peft>=0.19" bitsandbytes accelerate matplotlib
```

### Cell 2 — clone repo
```python
!rm -rf /kaggle/working/nano-gpt
!git clone https://github.com/olii-dev/nano-gpt.git /kaggle/working/nano-gpt
%cd /kaggle/working/nano-gpt
!git log -1 --oneline
```

> If you haven't pushed `pulse/kaggle_benchmark_pulse2.py` yet, push it first
> (`git add pulse/kaggle_benchmark_pulse2.py && git commit -m "Add Pulse 2 vs base benchmark" && git push`).

### Cell 3 — run
```python
!python -m pulse.kaggle_benchmark_pulse2 --modes with_system,no_system
```

~10–15 min total (loads each 4-bit model twice — once per mode). Outputs:

- `pulse/output/pulse2-vs-base.json` — full per-prompt results
- `pulse/output/pulse2-vs-base.png` — side-by-side bar chart

### Cell 4 — view chart
```python
from IPython.display import Image
Image("/kaggle/working/nano-gpt/pulse/output/pulse2-vs-base.png")
```

## How to read the result

- **`with_system`: Pulse 2 identity ≈ base identity** → the system prompt is doing the branding work, fine-tune adds little. *(This is what we expect based on the A/B test.)*
- **`with_system`: Pulse 2 identity > base identity** → the fine-tune genuinely helps branding on top of the prompt. Good.
- **`factual`: Pulse 2 ≥ base** → fine-tune did NOT cause catastrophic forgetting. Knowledge intact. Good.
- **`factual`: Pulse 2 < base** → fine-tune hurt knowledge. Bad — means identity data crowded out facts.
- **`no_system`: both fail identity** → confirms the fine-tune doesn't intrinsically own identity (the honest baseline).

The **bad_brands** count is the headline number for identity: how many identity prompts leaked an Alibaba/Qwen/ChatGPT mention.
