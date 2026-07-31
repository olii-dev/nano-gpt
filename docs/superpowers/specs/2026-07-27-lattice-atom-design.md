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
2. **Instruction-tuned model** — follows `### Instruction: ...` format, answers simple factual questions reliably (~90% on common facts like capitals). **Genuinely talkable** — see honest framing below.
3. **Benchmarked** — runs through MMLU/HellaSwag subset + custom eval, numbers compared to Mini, published honestly.

### The honest quality ceiling (read this twice)

**Atom at 100B tokens is genuinely talkable** — comparable to GPT-3 (not GPT-3.5/4). The binding constraint is still data, but at 100B tokens there's enough exposure for real learning:

- **SmolLM-360M** (HuggingFace's published model, same data family) trained on **600B tokens** — 6× more than Atom — so SmolLM will still be smarter on obscure facts.
- **Atom-100B** will reliably answer common factual questions (~90% on capitals, basic science), follow instructions, hold a 3-5 turn conversation, write coherent prose. Comparable to GPT-3 (2020, 125M on 300B tokens) — we have more params and ~⅓ the data, so similar tier.
- **Honest per-question projection:** "Capital of France?" → ~90% right. "Capital of Australia?" → reliably Canberra. "Explain photosynthesis" → fluent + mostly right. Multi-turn holds 3-5 turns before losing the thread. Obscure facts still hallucinated.

**This is the right call.** $375 of credits buys a genuinely conversational from-scratch model — a real published-small-model-tier artifact, not a toy. We keep ~$625 of credits for hosting + future work.

### Non-goals (explicit)

- **Not** as smart as Pulse 2 (Qwen3-8B saw ~18T tokens — 180× more).
- **Not** as smart as SmolLM-360M (it saw 600B tokens — 6× more). Atom will be weaker on obscure facts.
- **Not** a production assistant. Research demo, same framing as the rest of the Lattice lineup.
- **Going in:** comparable in *scale* (655M params) and *quality tier* (GPT-3-ish) to small published models. Real talkable model, honestly framed.

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

**Token budget: 100B tokens.** This is what buys "actually talkable." For context: SmolLM-360M was trained on 600B tokens (6× more) to reach its quality tier. At 100B tokens Atom lands in GPT-3 territory — genuinely conversational, reliably right on common facts, weaker on obscure ones. See the honest framing in §1.

**Data mix (Cosmopedia-tilted for small-model quality):**
- **Cosmopedia v2: 50B tokens** (synthetic textbooks/articles — *the* quality driver for small models per HuggingFace's findings)
- **FineWeb-Edu: 30B tokens** (real educational web)
- **Python-Edu: 20B tokens** (educational code)
- Total: **100B tokens**, ~500GB raw download
- **Tokenized form:** flat `.bin` of uint16 token IDs, ~200GB at 16k vocab
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

0.5% of SmolLM-Corpus (~500M tokens) held out as val set. Eval every 2000 steps, `best.pt` saved only on val improvement (same pattern as existing `train.py`).

---

## 4. Training strategy

### Phase 1: Pretraining (~250 GPU-hours, ~1 weekend on A100 — multiple VMs in parallel)

**The math:**
- 100B tokens ÷ 2048 seq len ≈ 50M training sequences
- Effective batch 512 (physical 64 × 8 grad-accum steps — fits easily on 80GB) → ~95,000 total steps
- A100 throughput on 655M: ~6-8 steps/min → **~230-260 GPU-hours pure training**
- Plus ~20h for data download + tokenization + debug runs → **~280h total wall-clock**

**How to fit 280h in a weekend (parallelize):**
A single A100 doing 280h = ~12 days. To hit a weekend, run **multiple A100 VMs in parallel**, each training on a data shard, then merge. Concretely: 4 × A100 VMs × ~70h each = done in ~3 days. Or accept ~10-12 days on a single VM (still well within credit budget — ~$420).

**Modern training tricks (the difference from Mini):**
- **bf16 mixed precision** — A100 has full bf16 support, 2× speedup, halves VRAM
- **Gradient accumulation** — effective batch 512 from physical 64
- **Cosine LR schedule + warmup** — warmup 2000 steps, decay to 10% of peak
- **Peak LR:** 3e-4 (standard for from-scratch pretraining at this scale)
- **Gradient clipping** (max norm 1.0) — stability, prevents loss spikes
- **Fused AdamW** (β1=0.9, β2=0.95, wd=0.1) — faster optimizer, standard for transformers

**Compute platform: Azure A100 80GB Spot VM(s).**

- **Spot preemption risk:** Azure can reclaim Spot VMs. For a long run, near-certain at least once.
- **Mitigation:** checkpoint to attached data disk every **30 minutes**. Auto-resume script detects existing checkpoint and continues. Worst case = 30 min of lost work per preemption.
- **Cost:** ~$1.50/hr Spot × ~250h training = **~$375 of Azure startup credits**. Uses ~⅓ of the ~$1000 balance. Leaves ~$625 for hosting + future work.
- **Escape hatch:** if Spot gets reclaimed repeatedly, flip to on-demand (~$3.20/hr → ~$130 total) — still cheap.

**Why not Kaggle here?** 100B tokens on a 655M model on T4 x2 = ~1000+ hours = months of multi-session checkpoint/resume hell. A100 collapses that to ~250h (parallelizable across VMs). The HF-push/pull-between-sessions complexity from earlier drafts is **gone** — one VM (or a small fleet), run to completion.

**Post-training:** push final checkpoint to HF (`oli-mebberson/lattice-atom-base`) for safekeeping before tearing down the VM.

### Phase 2: Instruction tuning (~4-6 GPU-hours, same VM or a fresh small one)

After pretraining completes (~step 95,000):

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
| 4 (Fri evening) | Spin up Azure A100 VM(s), attach 1TB data disk, download + tokenize SmolLM-Corpus (~500GB → ~200GB tokenized), launch pretrain | Pretrain running, first checkpoints saving |
| 4-14 | Pretrain runs to completion (~250h on one VM, or ~3 days on 4 parallel VMs), then instruction tune (~5h) | `lattice-atom-base` + `lattice-atom-instruct` |
| 15 | Push to HF, run benchmarks, add site card | Shipped |

**Total: ~3 days if parallelized across 4 VMs, ~12 days on a single VM. ~250h of A100 compute (~$375 of credits).**

The Mac days (1-3) are free and the real gating item — get the architecture + data pipeline proven locally before spending credits. The Azure run is then a known quantity (same pattern as Pulse 2).

---

## 8. Risks & honest mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Azure Spot VM preempted mid-run | High (long run) | Checkpoint every 30 min to attached data disk; auto-resume script. Worst case 30 min lost per preemption. If it happens >5 times, flip to on-demand (~$800 total — still within budget). |
| Model doesn't converge (loss spikes) | Low | Grad clipping, warmup, lower LR; sanity_test overfit batch catches wiring bugs before the big run |
| Atom feels "dumb" vs expectations | Low at 100B | 100B tokens is genuinely talkable territory (GPT-3 tier). SmolLM-360M will still beat it on obscure facts (6× more data) but Atom will reliably answer common questions. Honest framing in model card regardless. |
| Tokenizer regressed vs Mini's 8k | Low | Validate BPE on sample text before pretrain; fall back to 8k if compression is bad |
| Instruction tuning doesn't "take" | Low at 100B | Strong base → instruct should work. If not, publish base model honestly; it's still a legit talkable artifact. |
| Disk on A100 VM too small for 500GB data + 200GB tokens + checkpoints | Medium | Attach a 1TB managed data disk (~$10-15 total for the run). Same pattern as VM2 for Pulse 2, just bigger. |
| 250h single-VM run feels too long | Medium | Parallelize: 4 × A100 VMs each training on a data shard, then merge weights. Collapses wall-clock to ~3 days. More orchestration but well-understood pattern. |

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
3. **Exact SmolLM-Corpus shard selection:** which shards to hit 100B tokens? Sample across all three components (FineWeb-Edu / Cosmopedia / Python-Edu) for diversity, weight Cosmopedia higher per SmolLM's findings.
4. **Instruct data mix ratio:** Alpaca:OpenHermes:identity — settle during phase 2.
5. **Whether to add a 3rd live chat on the site** (Atom on free CPU/ZeroGPU) — decide after we see inference quality.
6. **Spot vs on-demand on Azure:** start Spot (~$375); flip to on-demand (~$800) if preemption happens >5 times. Still within the ~$1000 budget.
7. **Single-VM (~12 days) vs parallel fleet (~3 days):** decide based on how the first 10B tokens go. Parallel needs a shard-merge step at the end.

These are intentionally left for the implementation plan / runtime decisions, not the spec.
