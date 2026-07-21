# Benchmarking Lattice Pulse

Two layers: **what companies publish** vs **what you need for Pulse**.

## How big labs benchmark (public leaderboards)

They run **fixed test suites** on thousands of prompts with **automatic scoring** (not "vibes"):

| Suite | What it measures | Tool |
|-------|------------------|------|
| **MMLU** | Knowledge (57 subjects) | [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| **HellaSwag / ARC** | Commonsense | lm-eval-harness |
| **GSM8K** | Math word problems | lm-eval-harness |
| **MT-Bench** | Chat quality (GPT judge) | [FastChat](https://github.com/lm-sys/FastChat) |
| **HELM** | Broad scenarios | Stanford HELM |

Example (compare Pulse vs base Qwen on MMLU — slow, ~30–60 min on GPU):

```bash
pip install lm-eval

# Base model
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct \
  --tasks mmlu --device cuda --batch_size 4

# Pulse (local or HF id)
lm_eval --model hf \
  --model_args pretrained=oli-mebberson/lattice-pulse \
  --tasks mmlu --device cuda --batch_size 4
```

Publish numbers like: `MMLU: 62.1%` (fine-tune usually trades a little MMLU for better chat/branding).

## What Lattice needs (custom eval)

Public benchmarks won't measure **"says Lattice Pulse not SmolLM"**. Use:

```bash
# Our suite: identity + facts + multi-turn
python -m pulse.benchmark --model pulse/output/lattice-pulse

# Compare Pulse vs base Qwen side-by-side
python -m pulse.benchmark --compare pulse,base

# Compare named checkpoints
python -m pulse.benchmark --compare \
  pulse=pulse/output/lattice-pulse \
  base=Qwen/Qwen2.5-1.5B-Instruct \
  hf=oli-mebberson/lattice-pulse
```

Output: `pulse/output/benchmark-compare.json` + pass-rate table.

**Run on Kaggle GPU** for stable branding scores (Mac MPS is noisier):

```python
!python -m pulse.benchmark --model oli-mebberson/lattice-pulse --device cuda \
  --json-out pulse/output/benchmark-kaggle.json
```

## Suggested workflow

1. **Custom suite** (`pulse.benchmark`) — before/after every fine-tune
2. **lm-eval MMLU + GSM8K** — once per major release (prove you didn't break the model)
3. **Manual spot-check** — 5 min chat on pulse.html after deploy
