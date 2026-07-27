# Lattice Atom — Design Spec

**Date:** 2026-07-27
**Status:** Approved (pending spec review)
**Author:** Oli Mebberson

---

## 1. What we're building

**Lattice Atom** — a ~350M parameter GPT-style language model, **trained entirely from scratch** (every parameter learned by us from random initialization), then instruction-tuned so it can follow basic instructions and answer simple questions.

This is the flagship from-scratch model in the Lattice lineup:

| Model | Params | Origin | Role |
|-------|--------|--------|------|
| Lattice Mini | 42M | From scratch | Original learning project |
| **Lattice Atom** | **~350M** | **From scratch** | **Flagship — this spec** |
| Lattice Air | 148M | From scratch | Parked (R&D, mode-collapsed) |
| Lattice Pulse | 1.5B | Fine-tune of Qwen2.5 | Conversational demo |
| Lattice Pulse 2 | 8B | QLoRA on Qwen3-8B | Smarter demo (research) |

**The defining property:** Atom is a *real standalone model*. Not a LoRA, not an adapter, not a fine-tune of someone else's weights. All ~350M params are trained by us. The final artifact is one self-contained checkpoint loadable with plain `from_pretrained` — no `peft`, no base model dependency.

### Goals (definition of done)

1. **Pretrained base model** — coherent English generation, val loss meaningfully below Mini's. Demoable: "write about X" produces fluent paragraphs.
2. **Instruction-tuned model** — follows `### Instruction: ...` format, answers simple factual questions recognizably (e.g. "capital of France" → "Paris"). Genuinely "talk-to-able," not just autocomplete.
3. **Benchmarked** — runs through MMLU/HellaSwag subset + custom eval, numbers compared to Mini, published honestly.

### Non-goals (explicit)

- **Not** as smart as Pulse 2. 5B training tokens vs Qwen3-8B's ~18 trillion. Atom will be coherent but not knowledgeable.
- **Not** a production assistant. Research demo, same framing as the rest of the Lattice lineup.
- **Not** going to beat published small models (SmolLM, Qwen2.5-0.5B). Those had more compute + data.

---

## 2. Architecture

GPT-style decoder-only transformer. Modern (2023-era) architecture — same family as Llama/Qwen, not the 2018-era tricks Mini used.

| Component | Mini (existing) | **Atom (new)** | Why |
|-----------|-----------------|----------------|-----|
| Layers | 12 | **24** | More depth = better reasoning |
| Embed dim | 512 | **1024** | Wider = more capacity |
| Heads | 8 | **16** | More attention heads |
| Context | 512 | **1024** | Longer context for instruction-following |
| Vocab | 8k | **16k** (BPE, retrained on FineWeb-Edu) | Better text coverage |
| **Params** | **42M** | **~350M** | 8× bigger |
| Positional encoding | Absolute learned | **RoPE** | Generalizes beyond trained length, modern |
| Normalization | LayerNorm | **RMSNorm** | Faster, better, used by Llama/Qwen |
| FFN activation | GELU | **SwiGLU** | Better gradient flow, modern standard |
| Attention | Manual | **Manual (custom impl)** | Educational, ours — not importing SDPA blackbox |
| Tie embeddings | No | **Yes** (input/output) | Saves params at this scale, standard |

**Implementation:** extends the existing `model.py` (which already has `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `GPT`). Upgrades each component to the modern variant. All code remains ours — no importing a pretrained architecture.

**Memory check (Kaggle T4 x2, 32GB total):**
- Weights (bf16): ~700MB
- AdamW state (fp32, 2 moments): ~2.8GB
- Activations @ batch 32 × seq 1024: ~4GB
- **Total ~8GB — comfortable headroom on 32GB**

---

## 3. Data

### Pretraining corpus

**FineWeb-Edu** (HuggingFace, open, free) — curated educational web content filtered by a quality classifier. This is what real labs use for small-model pretraining (HuggingFace's own SmolLM trained on a FineWeb-Edu-heavy mix).

- **Token budget: 5B tokens** (Chinchilla-ish optimal for 350M params; target ~14 tokens/param)
- **Source:** ~50 FineWeb-Edu parquet shards (~100M tokens each, ~500MB each)
- **Download size:** ~25GB raw
- **Tokenized form:** flat `.bin` of uint16 token IDs, ~10GB for 5B tokens at 16k vocab
- **Epochs:** ~1 (don't overfit, see data once or twice)

### Tokenizer

**Retrain BPE on FineWeb-Edu sample** — new 16k vocab trained on a ~500M-token subsample. Stored in `tokenizer/atom/`. Per-dataset tokenizer dirs (existing pattern in `config.py`).

### Instruction tuning corpus (phase 2)

~50k instruction/response pairs:
- **Alpaca** (Stanford, clean, 52k) — primary, subset to ~40k
- **OpenHermes slice** (~5k higher-quality examples) — quality boost
- **Lattice identity examples** (~500 hand-written) — own the brand from scratch this time
- **Format:** `### Instruction:\n{q}\n\n### Response:\n{a}` (reuse existing `instruct_format.py`)

### Validation split

0.5% of FineWeb-Edu (~25M tokens) held out as val set. Eval every 500 steps, `best.pt` saved only on val improvement (same pattern as existing `train.py`).

---

## 4. Training strategy

### Phase 1: Pretraining (~100 GPU-hours, ~3-4 weeks of Kaggle)

**The math:**
- 5B tokens ÷ 1024 seq len ≈ 5M training sequences
- Effective batch 256 (physical 32 × 8 grad-accum steps) → ~19,500 total steps
- T4 x2 throughput on 350M: ~3-4 steps/min → ~100 GPU-hours
- Kaggle quota: 30h/week → **~3-4 Save & Run All sessions**

**Modern training tricks (the difference from Mini):**
- **bf16 mixed precision** — T4 supports it, 2× speedup, halves VRAM
- **Gradient accumulation** — effective batch 256 from physical 32
- **Cosine LR schedule + warmup** — warmup 2000 steps, decay to 10% of peak
- **Peak LR:** 3e-4 (standard for from-scratch pretraining at this scale)
- **Gradient clipping** (max norm 1.0) — stability, prevents loss spikes
- **Fused AdamW** (β1=0.9, β2=0.95, wd=0.1) — faster optimizer, standard for transformers

**Checkpoint/resume strategy (critical):**

Kaggle's `/kaggle/working/` does NOT persist between Save & Run All sessions — each version starts fresh. We use **HuggingFace as checkpoint store**:

```
Session start:  pull latest checkpoint from private HF repo
                 lattice-atom-training (model + optimizer state + step)
Session runs:   ~12h of training, checkpoint every 500 steps to /kaggle/working/
Session end:    push latest checkpoint + optimizer state back to HF repo
Next session:   pulls and resumes from saved step
```

- ~1.4GB upload/download per session (weights + optimizer in fp32 = ~4GB; we'll explore bf16 optimizer state to halve this)
- Persists forever — if Kaggle dies entirely, no work lost
- Same `maybe_save_best()` pattern from existing `train.py`, extended to also push to HF

**Why not Kaggle Dataset as checkpoint store?** HF is cleaner — script does it automatically with `hf_hub`, no manual dataset upload clicks between sessions.

### Phase 2: Instruction tuning (~6-8 GPU-hours, single session)

After pretraining completes (~step 19,500):

- **Method: full fine-tune** (not LoRA — we own all 350M params, tune them all)
- **LR:** 1e-5 (refining, not learning from scratch — 30× lower than pretrain)
- **Epochs:** 2-3 over the 50k examples → ~150k examples seen
- **Output:** `lattice-atom-instruct.pt`
- **Single Kaggle session** — fits in 12h easily

---

## 5. File structure (extends existing repo)

```
nano-gpt/
├── model.py              ← UPGRADED: RoPE, RMSNorm, SwiGLU, tied embeddings
├── config.py             ← UPGRADED: lattice_atom_config() preset
├── tokenizer.py          ← unchanged (retrains BPE per-dataset, already supports this)
├── dataset.py            ← UPGRADED: FineWeb-Edu in DATASET_REGISTRY + streaming
├── train.py              ← UPGRADED: bf16, grad-accum, cosine LR, HF checkpoint push/pull
├── finetune.py           ← unchanged pattern (Alpaca-style, reuse for Atom instruct)
├── generate.py           ← unchanged (LMEngine, --instruct / --base modes)
├── sanity_test.py        ← UPGRADED: overfit-on-tiny-batch test for new arch
├── atom/                 ← NEW: phase-specific scripts
│   ├── prepare_fineweb.py   ← download + tokenize FineWeb-Edu → flat .bin
│   ├── hf_checkpoint.py     ← push/pull training snapshots to private HF repo
│   └── KAGGLE_ATOM.md       ← run instructions (mirrors KAGGLE_PULSE2.md)
├── instruct_format.py    ← unchanged
└── requirements.txt      ← UPGRADED: transformers only needed for tokenizers, not the model
```

**Key principle:** this extends the existing Mini/Air infrastructure, not a new repo. Same `train.py`, same patterns, just bigger + modernized. Mini still works as a regression test.

---

## 6. Deployment & hosting

After training completes:

**HuggingFace:**
- `oli-mebberson/lattice-atom-base` — raw pretrained weights, apache-2.0
- `oli-mebberson/lattice-atom-instruct` — instruction-tuned, apache-2.0
- Both fully standalone, honest model cards (same pattern as Pulse 2's)

**Site:**
- New card on `index.html` between Mini and Pulse: "Lattice Atom · 350M · From scratch"
- Optional: dedicated chat page `atom.html` if we want a 3rd live model (350M runs on free CPU/ZeroGPU, no Azure VM needed — that's the nice thing about small models)
- Benchmarks added to `/benchmarks` page comparing Atom vs Mini (and vs published SmolLM-360M if we want to be brave)

**Compute:** Atom is small enough (350M) to run on free hosting — HuggingFace ZeroGPU Space or even CPU. No Azure credits burned for inference, unlike Pulse 2.

---

## 7. Timeline

| Week | Phase | Output |
|------|-------|--------|
| 1 | Infrastructure: upgrade `model.py`/`config.py`, write `prepare_fineweb.py`, validate with overfit test, train tokenizer | Working pipeline, first tokens generated |
| 2-3 | Pretraining session 1-2 (HF checkpoint pull/push working) | ~step 10,000, val loss dropping |
| 4 | Pretraining session 3 (finish ~step 19,500) + instruction tuning | `lattice-atom-base` + `lattice-atom-instruct` |
| 4 | Benchmarks + HF upload + site card | Shipped |

**Total: ~4 weeks of elapsed time, ~100 GPU-hours + ~8 GPU-hours (all on free Kaggle quota).**

---

## 8. Risks & honest mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Kaggle quota runs out mid-pretrain | Medium | HF checkpoints mean we lose ≤12h max; can fall back to Lightning AI's 80h free or Mac MPS for a session |
| Model doesn't converge (loss spikes) | Low | Grad clipping, warmup, lower LR; sanity_test overfit batch catches wiring bugs before the big run |
| Atom feels "dumb" vs expectations | High (honest) | This is **expected** — 5B tokens ≠ 18T. Definition of done is "coherent + recognizably answers simple questions," not "smart like Pulse 2." Model card will frame honestly. |
| Tokenizer regressed vs Mini's 8k | Low | Validate BPE on sample text before pretrain; fall back to 8k if compression is bad |
| Instruction tuning doesn't "take" | Medium | If base is too weak, instruct version is incoherent — that's a real possible outcome. Mitigation: publish base model honestly even if instruct fails; the base is still a legit artifact. |

---

## 9. What success looks like

**Minimum viable success:**
- A pretrained 350M model that generates coherent English paragraphs
- Val loss meaningfully below Mini's
- Uploaded to HF with an honest model card

**Full success:**
- All of the above PLUS an instruction-tuned version that answers `### Instruction: What is the capital of France?` with something recognizably close to "Paris"
- Benchmarks on `/benchmarks` page comparing Atom vs Mini
- A third live chat model on the site (runs on free hosting)

**The honest framing for whatever we ship:** Atom is a from-scratch model trained on 5B tokens. It is coherent but not knowledgeable. It is fully ours — every parameter learned by us. That's the point.

---

## Open questions for implementation plan

1. **bf16 vs fp16 on T4:** T4 nominally supports bf16 but with caveats; fp16 is safer. Decide during sanity test.
2. **Optimizer state precision:** fp32 (4GB checkpoint) vs bf16 (2GB). Test if bf16 optimizer state hurts convergence.
3. **Exact FineWeb-Edu shard selection:** which 50 shards? Random sample or first 50? (Should sample for diversity.)
4. **Instruct data mix ratio:** Alpaca:OpenHermes:identity — settle during phase 2.
5. **Whether to add a 3rd live chat on the site** (Atom on free CPU/ZeroGPU) — decide after we see inference quality.

These are intentionally left for the implementation plan / runtime decisions, not the spec.
