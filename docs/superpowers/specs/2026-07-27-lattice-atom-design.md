# Lattice Atom — Design Spec

**Date:** 2026-07-27
**Status:** Approved (pending spec review)
**Author:** Oli Mebberson

---

## 1. What we're building

**Lattice Atom** — a ~655M parameter GPT-style language model, **trained entirely from scratch** (every parameter learned by us from random initialization), then instruction-tuned so it can follow basic instructions and answer simple questions.

This is the flagship from-scratch model in the Lattice lineup:

| Model | Params | Origin | Role |
|-------|--------|--------|------|
| Lattice Mini | 42M | From scratch | Original learning project |
| **Lattice Atom** | **~655M** | **From scratch** | **Flagship — this spec** |
| Lattice Air | 148M | From scratch | Parked (R&D, mode-collapsed) |
| Lattice Pulse | 1.5B | Fine-tune of Qwen2.5 | Conversational demo |
| Lattice Pulse 2 | 8B | QLoRA on Qwen3-8B | Smarter demo (research) |

**The defining property:** Atom is a *real standalone model*. Not a LoRA, not an adapter, not a fine-tune of someone else's weights. All ~655M params are trained by us. The final artifact is one self-contained checkpoint loadable with plain `from_pretrained` — no `peft`, no base model dependency.

### Goals (definition of done)

1. **Pretrained base model** — coherent English generation, val loss meaningfully below Mini's. Demoable: "write about X" produces fluent paragraphs.
2. **Instruction-tuned model** — follows `### Instruction: ...` format, gives *recognizable* answers to simple questions. **"Barely talkable"** — see honest framing below.
3. **Benchmarked** — runs through MMLU/HellaSwag subset + custom eval, numbers compared to Mini, published honestly.

### The honest quality ceiling (read this twice)

**Atom at 10B tokens will be "barely talkable," not a real assistant.** The binding constraint is data, not architecture or compute budget:

- **SmolLM-360M** (HuggingFace's published model, same data we're using) trained on **600B tokens** — 60× more than Atom — to reach genuinely conversational quality.
- **Atom-10B** will produce fluent prose and follow the instruction format, but factual answers will mostly be wrong or sketchy (each fact seen ~1 time). Think **GPT-1 (2018, 117M) vibes, not GPT-3.**
- Reaching real SmolLM-tier chat would need ~600B tokens = ~$2,200 of compute. We have ~$1,000 of credits total, so even spending everything caps at ~250B tokens / GPT-3-mini-tier.

**This is intentional.** Atom is a from-scratch learning artifact at honest scale, not an attempt to compete with published models. The value is owning every parameter and the full pipeline, not output quality.

### Non-goals (explicit)

- **Not** as smart as Pulse 2 (Qwen3-8B saw ~18T tokens).
- **Not** comparable to SmolLM-360M in quality (it saw 60× more data).
- **Not** a production assistant. Research demo, same framing as the rest of the Lattice lineup.
- **Going in:** comparable in *scale* (655M params) to small published models, but **not** in quality (10B vs 600B+ tokens).

---

## 2. Architecture

GPT-style decoder-only transformer. Modern (2023-era) architecture — same family as Llama/Qwen, not the 2018-era tricks Mini used.

| Component | Mini (existing) | **Atom (new)** | Why |
|-----------|-----------------|----------------|-----|
| Layers | 12 | **32** | More depth = better reasoning |
| Embed dim | 512 | **1280** | Wider = more capacity |
| Heads | 8 | **20** | More attention heads |
| Context | 512 | **2048** | Longer context for instruction-following |
| Vocab | 8k | **16k** (BPE, retrained on SmolLM-Corpus sample) | Better text coverage |
| **Params** | **42M** | **~655M** | 15× bigger |
| Positional encoding | Absolute learned | **RoPE** | Generalizes beyond trained length, modern |
| Normalization | LayerNorm | **RMSNorm** | Faster, better, used by Llama/Qwen |
| FFN activation | GELU | **SwiGLU** (hidden 3456) | Better gradient flow, modern standard |
| Attention | Manual | **Manual (custom impl)** | Educational, ours — not importing SDPA blackbox |
| Tie embeddings | No | **Yes** (input/output) | Saves params at this scale, standard |

**Implementation:** extends the existing `model.py` (which already has `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `GPT`). Upgrades each component to the modern variant. All code remains ours — no importing a pretrained architecture.

**Param math (verified):** 32 layers × (4·1280² attention + 3·1280·3456 SwiGLU FFN) + 16k·1280 tied embeddings = **654,868,480 ≈ 655M**.

**Memory check (A100 80GB):**
- Weights (bf16): ~1.31 GB
- AdamW state (fp32, 2 moments): ~5.24 GB
- Activations @ batch 32 × seq 2048: ~5.4 GB
- **Total ~12 GB — massive headroom on 80GB** (lets us crank batch size way up)

---

## 3. Data

### Pretraining corpus

**SmolLM-Corpus** (HuggingFace, open, free) — the exact dataset HuggingFace built and used to train SmolLM. Purpose-built for small-model pretraining. Three components:

1. **FineWeb-Edu** — curated educational web content (~5B token slice)
2. **Cosmopedia v2** — synthetic textbooks and articles generated by larger models. *This is the secret sauce* — known to give small models a big quality boost because the text is clean, structured, and information-dense.
3. **Python-Edu** — educational Python code (~800M token slice)

We're replicating HuggingFace's published recipe rather than inventing our own data mix — that's the highest-quality, lowest-risk choice for a small model.

**Token budget: 10B tokens.** This is the binding constraint of the project. For context: SmolLM-360M was trained on 600B tokens (60× more) to reach genuinely talkable quality. At 10B tokens Atom will be **"barely talkable"** — see the honest framing in §1.

**Data mix (Cosmopedia-tilted for small-model quality):**
- **Cosmopedia v2: 5B tokens** (synthetic textbooks/articles — *the* quality driver for small models per HuggingFace's findings)
- **FineWeb-Edu: 3B tokens** (real educational web)
- **Python-Edu: 2B tokens** (educational code)
- Total: **10B tokens**, ~50GB raw download
- **Tokenized form:** flat `.bin` of uint16 token IDs, ~20GB at 16k vocab
- **Epochs:** ~1 (see data once)

The Cosmopedia tilt is deliberate — HuggingFace's published ablation shows small models learn disproportionately from clean structured text vs raw web. Same total tokens, better result.

### Tokenizer

**Retrain BPE on SmolLM-Corpus sample** — new 16k vocab trained on a ~500M-token subsample. Stored in `tokenizer/atom/`. Per-dataset tokenizer dirs (existing pattern in `config.py`).

### Instruction tuning corpus (phase 2)

~50k instruction/response pairs:
- **Alpaca** (Stanford, clean, 52k) — primary, subset to ~40k
- **OpenHermes slice** (~5k higher-quality examples) — quality boost
- **Lattice identity examples** (~500 hand-written) — own the brand from scratch this time
- **Format:** `### Instruction:\n{q}\n\n### Response:\n{a}` (reuse existing `instruct_format.py`)

### Validation split

0.5% of SmolLM-Corpus (~50M tokens) held out as val set. Eval every 1000 steps, `best.pt` saved only on val improvement (same pattern as existing `train.py`).

---

## 4. Training strategy

### Phase 1: Pretraining (~40 GPU-hours, ~1 weekend on A100)

**The math:**
- 10B tokens ÷ 2048 seq len ≈ 5M training sequences
- Effective batch 512 (physical 64 × 8 grad-accum steps — fits easily on 80GB) → ~9,500 total steps
- A100 throughput on 655M: ~6-8 steps/min → **~20-25 GPU-hours pure training**
- Plus ~15h for data download + tokenization + debug runs → **~40h total wall-clock, one weekend**

**Modern training tricks (the difference from Mini):**
- **bf16 mixed precision** — A100 has full bf16 support, 2× speedup, halves VRAM
- **Gradient accumulation** — effective batch 512 from physical 64
- **Cosine LR schedule + warmup** — warmup 1000 steps, decay to 10% of peak
- **Peak LR:** 3e-4 (standard for from-scratch pretraining at this scale)
- **Gradient clipping** (max norm 1.0) — stability, prevents loss spikes
- **Fused AdamW** (β1=0.9, β2=0.95, wd=0.1) — faster optimizer, standard for transformers

**Compute platform: Azure A100 80GB Spot VM.** Single VM, single continuous run over a weekend.

- **Spot preemption risk:** Azure can reclaim Spot VMs. For a ~25h run, real possibility.
- **Mitigation:** checkpoint to local disk (`/mnt/azure-volume` or attached data disk) every **30 minutes**. Auto-resume script detects existing checkpoint and continues. Worst case = 30 min of lost work per preemption.
- **Cost:** ~$1.50/hr Spot × ~40h wall-clock = **~$60 of Azure startup credits**. Trivial vs the ~$1000 balance.
- **Escape hatch:** if Spot gets reclaimed repeatedly, flip to on-demand (~$3.20/hr → ~$130 total) — still cheap.

**Why not Kaggle here?** 10B tokens on a 655M model on T4 x2 = ~100+ hours = 3-4 weeks of multi-session checkpoint/resume hell. A100 collapses that to one weekend. The HF-push/pull-between-sessions complexity from earlier drafts is **gone** — one VM, run to completion, done.

**Post-training:** push final checkpoint to HF (`oli-mebberson/lattice-atom-base`) for safekeeping before tearing down the VM.

### Phase 2: Instruction tuning (~4-6 GPU-hours, same VM or a fresh small one)

After pretraining completes (~step 9,500):

- **Method: full fine-tune** (not LoRA — we own all 655M params, tune them all)
- **LR:** 1e-5 (refining, not learning from scratch — 30× lower than pretrain)
- **Epochs:** 2-3 over the 50k examples → ~150k examples seen
- **Output:** `lattice-atom-instruct.pt`
- **Same A100 session** right after pretrain, or a fresh cheap VM (A10, ~$5)

---

## 5. File structure (extends existing repo)

```
nano-gpt/
├── model.py              ← UPGRADED: RoPE, RMSNorm, SwiGLU, tied embeddings
├── config.py             ← UPGRADED: lattice_atom_config() preset
├── tokenizer.py          ← unchanged (retrains BPE per-dataset, already supports this)
├── dataset.py            ← UPGRADED: SmolLM-Corpus in DATASET_REGISTRY + streaming
├── train.py              ← UPGRADED: bf16, grad-accum, cosine LR, 30-min local checkpoint + auto-resume
├── finetune.py           ← unchanged pattern (Alpaca-style, reuse for Atom instruct)
├── generate.py           ← unchanged (LMEngine, --instruct / --base modes)
├── sanity_test.py        ← UPGRADED: overfit-on-tiny-batch test for new arch
├── atom/                 ← NEW: phase-specific scripts
│   ├── prepare_data.py       ← download + tokenize SmolLM-Corpus → flat .bin
│   ├── azure_train.sh        ← launch pretrain on A100 VM with checkpoint/resume
│   └── AZURE_ATOM.md         ← VM setup + run instructions
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
- New card on `index.html` between Mini and Pulse: "Lattice Atom · 655M · From scratch"
- Optional: dedicated chat page `atom.html` if we want a 3rd live model (655M in 4-bit quantization = ~400MB, runs on free CPU/ZeroGPU, no Azure VM needed for inference — that's the nice thing about small models)
- Benchmarks added to `/benchmarks` page comparing Atom vs Mini (and vs published SmolLM-360M if we want to be brave)

**Compute for inference:** Atom is small enough (655M, ~400MB in 4-bit) to run on free hosting — HuggingFace ZeroGPU Space or even CPU. No Azure credits burned for inference, unlike Pulse 2.

---

## 7. Timeline

| Day | Phase | Output |
|-----|-------|--------|
| 1-3 (Mac, local) | Infrastructure: upgrade `model.py`/`config.py`, write `prepare_data.py`, validate with overfit test, train tokenizer on Mac | Working pipeline, first tokens generated, sanity-checked |
| 4 (Fri evening) | Spin up Azure A100 VM, download + tokenize SmolLM-Corpus (~50GB → ~20GB tokenized), launch pretrain | Pretrain running, first checkpoints saving |
| 4-5 (weekend) | Pretrain runs to completion (~25h), then instruction tune (~5h) | `lattice-atom-base` + `lattice-atom-instruct` |
| 6 | Push to HF, run benchmarks, add site card | Shipped |

**Total: ~6 days elapsed, ~30h of A100 compute (~$45-60 of credits).**

The infrastructure days (1-3) can be shorter if you push hard — the real commitment is the weekend where the VM runs.

---

## 8. Risks & honest mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Azure Spot VM preempted mid-run | Medium | Checkpoint every 30 min locally; auto-resume script. Worst case 30 min lost. If it happens >3 times, flip to on-demand (~$130 total, still cheap). |
| Model doesn't converge (loss spikes) | Low | Grad clipping, warmup, lower LR; sanity_test overfit batch catches wiring bugs before the big run |
| Atom feels "dumb" vs expectations | Medium (honest) | 10B tokens is real small-model territory (SmolLM-360M used 2T — we have 200× less). Definition of done is "coherent + recognizably answers simple questions," not "smart like Pulse 2." Model card will frame honestly. |
| Tokenizer regressed vs Mini's 8k | Low | Validate BPE on sample text before pretrain; fall back to 8k if compression is bad |
| Instruction tuning doesn't "take" | Medium | If base is too weak, instruct version is incoherent. Mitigation: publish base model honestly even if instruct fails; the base is still a legit artifact. |
| Disk on A100 VM too small for 50GB data + 20GB tokens + checkpoints | Low | Attach a 256GB managed data disk (~$2-3 total for a weekend). Same pattern as VM2 for Pulse 2. |

---

## 9. What success looks like

**Minimum viable success:**
- A pretrained 655M model that generates coherent English paragraphs
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
3. **Exact SmolLM-Corpus shard selection:** which shards to hit 10B tokens? Sample across all three components (FineWeb-Edu / Cosmopedia / Python-Edu) for diversity, weight Cosmopedia higher per SmolLM's findings.
4. **Instruct data mix ratio:** Alpaca:OpenHermes:identity — settle during phase 2.
5. **Whether to add a 3rd live chat on the site** (Atom on free CPU/ZeroGPU) — decide after we see inference quality.
6. **Spot vs on-demand on Azure:** start Spot (~$60); flip to on-demand (~$130) if preemption happens >3 times.

These are intentionally left for the implementation plan / runtime decisions, not the spec.
