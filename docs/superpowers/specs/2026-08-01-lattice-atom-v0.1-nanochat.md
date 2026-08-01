# Lattice Atom v0.1 — nanochat-based design spec

**Date:** 2026-08-01
**Status:** Approved (pending spec review)
**Supersedes:** `2026-07-27-lattice-atom-design.md` (the custom-stack spec — kept in git as a learning artifact, not the active plan)
**Author:** Oli Mebberson

---

## 1. What we're building

**Lattice Atom v0.1** — a ~160M parameter GPT-style language model, trained from scratch on SmolLM-Corpus using [Karpathy's nanochat](https://github.com/karpathy/nanochat) as the training framework, then Lattice-branded via SFT. A real, publishable, talkable from-scratch model for ~$111.

This is the same project goal as the earlier custom-stack spec, but the infrastructure decision has changed: **we use nanochat instead of maintaining our own training stack.** The Atom identity is preserved through four things we control — tokenizer, data, SFT, and naming — not through a forked architecture.

### Lineup placement

| Model | Params | Origin | Role |
|-------|--------|--------|------|
| Lattice Mini | 42M | From scratch (custom) | Original learning project |
| **Lattice Atom v0.1** | **~160M** | **From scratch (nanochat)** | **This spec — talkable from-scratch** |
| Lattice Pulse | 1.5B | Fine-tune of Qwen2.5 | Conversational demo |
| Lattice Pulse 2 | 8B | QLoRA on Qwen3-8B | Smarter demo (research) |

Atom is the from-scratch flagship: bigger and more useful than Mini, fully ours (every parameter trained from random init by us), and genuinely chatty after the full nanochat pipeline.

### Goals (definition of done)

1. **Pretrained base model** — coherent English generation with modern small-model conversational capability (100–200M parameter class). 160M trained at 260 tokens/parameter, solidly overtrained per SmolLM's small-model thesis.
2. **Instruction-tuned model** — follows chat format, holds a short conversation, answers common factual questions recognizably.
3. **Optionally: RL-tuned model** — if nanochat's `chat_rl.py` (GRPO) lands cleanly, a third variant with sharper responses. (Stretch goal, not blocking.)
4. **Benchmarked** — runs through nanochat's eval (DCLM CORE, bits/byte), numbers compared honestly to Mini and to published small-model numbers.
5. **Published** — base + instruct on HuggingFace as `oli-mebberson/lattice-atom-base` and `oli-mebberson/lattice-atom-instruct`, honest model cards, MIT license (matching nanochat).

### Release criteria for v0.1 (all must pass to ship)

A checklist, not vibes. v0.1 ships only when every box is ticked:

- [ ] **RC1 — Training complete:** pretrain ran to completion (loss converged, no NaNs), checkpoint saved. SFT ran to completion. (RL optional — see RC-stretch below.)
- [ ] **RC2 — Evals run and recorded:** nanochat's `base_eval.py` and `chat_eval.py` both executed; DCLM CORE score + bits-per-byte captured for the base model; chat eval captured for the instruct model. Numbers written to the model card and the `/benchmarks` page.
- [ ] **RC3 — SFT chat quality gate:** the instruct model, tested with NO system prompt, can (a) follow a `### Instruction:`-style prompt, (b) answer "What is the capital of France?" with a response containing "Paris," (c) answer "Who are you?" with a response containing "Lattice." If any of these fail, SFT needs more data or steps before shipping.
- [ ] **RC4 — Lattice identity measurable:** the identity eval from §6 passes (see that section for the bar).
- [ ] **RC5 — Model card honest + complete:** HuggingFace model card states param count, training data, FLOPs, eval numbers, the nanochat base, MIT license, and an honest "modern small-model class, not frontier" framing. No overselling.
- [ ] **RC6 — HuggingFace release:** `lattice-atom-base` and `lattice-atom-instruct` pushed, public, downloadable, loadable with a standard `from_pretrained` call (verified by loading on Mac).
- [ ] **RC-stretch — RL variant (optional):** if GRPO ran and improved chat quality measurably, publish `lattice-atom-rl`. Not blocking for v0.1.

### Non-goals (explicit, v0.1)

- **Not** modifying nanochat's architecture. Run it unmodified first; SwiGLU/GQA experiments are v0.2.
- **Not** 655M. That's a v0.2 scale-up once the pipeline is proven. (The 655M @ 35B-token plan from the old spec is parked — its 53 tokens/param ratio was starving the model.)
- **Not** competing with frontier models. Atom v0.1 targets modern small-model conversational capability (100–200M class), not GPT-4-tier or even SmolLM-360M-tier quality.
- **Not** a production assistant. Research demo, same framing as the rest of the Lattice lineup.

### Honest quality expectation

**Modern small-model conversational capability (100–200M parameter class), not frontier.** 160M params trained at ~260 tokens/param (vs GPT-3 125M at 2400:1, SmolLM-135M at 4444:1). Each parameter is well-fed; the model will produce coherent English, follow chat format, and answer simple questions. It sits in the same capability class as SmolLM-135M / early small open models — genuinely conversational within its size tier, not competing with larger or frontier models. The value is: every parameter trained by us, full pipeline owned, ~$111 total cost.

---

## 2. Why nanochat (the migration rationale)

The earlier spec built a custom training stack (model.py + train.py + dataset.py + checkpoint logic). That code is in git history as a learning artifact. **We're not using it for the real run because nanochat solves problems we'd be solving for the first time on a $382 gamble.**

| Problem | Custom stack | nanochat |
|---------|-------------|----------|
| Multi-GPU / distributed | ❌ single-VM only | ✅ torchrun, FSDP, 8×A100 proven |
| Spot preemption recovery | untested at scale | ✅ `checkpoint_manager.py` (Karpathy's) |
| Flash Attention 3 | ❌ SDPA only | ✅ native + SDPA fallback |
| Eval (DCLM CORE, bpb) | ❌ we'd build it | ✅ `base_eval.py`, `chat_eval.py` |
| RL phase (GRPO) | ❌ not planned | ✅ `chat_rl.py` |
| First-run risk | high ($382, untested) | low ($111, Karpathy-proven) |

The full-pipeline learning goal is also better served: nanochat gives us pretrain → SFT → **RL** → eval, where the custom plan had only pretrain → SFT. Skipping RL was a real quality gap.

### What we keep from the old work

Nothing is deleted. The custom AtomGPT (RoPE + RMSNorm + SwiGLU + tied embeddings) lives in git at commits `7863c8d`–`62559a1` as a learning artifact with passing tests. The SwiGLU design specifically is worth revisiting in v0.2 (SwiGLU is the modern standard; nanochat uses ReLU² which is experimental). But we don't fork nanochat before we've ever run it.

### What carries over as *our* choices (not infrastructure)

1. **Data: SmolLM-Corpus, Cosmopedia-tilted** — HuggingFace's small-model recipe. We point nanochat at it instead of its default ClimbMix.
2. **Tokenizer: trained on SmolLM-Corpus** — our vocab/data decision, nanochat's tokenizer training code.
3. **Lattice identity SFT data** — the brand layer that makes Atom *Lattice* Atom, not "nanochat with our name on it."
4. **Naming + framing** — Atom in the Lattice lineup, honest model cards.

---

## 3. Architecture (nanochat's, unmodified)

We use nanochat's `gpt.py` as-is for v0.1. For reference, it's already modern:

| Component | nanochat | Notes |
|-----------|----------|-------|
| Position | RoPE | Same as our AtomGPT |
| Norm | RMSNorm (param-less, `F.rms_norm`) | Slightly different from ours (we had learnable scale) |
| FFN | ReLU² | Experimental; SwiGLU is the standard. **v0.2 swap candidate.** |
| Attention | Flash Attention 3 + GQA | Better than our SDPA-only impl |
| Embeddings | Untied | Different from our tied choice |
| Init | Karpathy's | Proven |

**Model size:** nanochat's depth dial — GPT-2-equivalent is depth ~26 (~160M params). That's the v0.1 target. The depth param scales cleanly for v0.2 if we want bigger.

**Why not fork for SwiGLU now:** running upstream first proves the pipeline end-to-end on a $111 run. If we fork before the first run and hit a bug, we can't tell whether it's our SwiGLU port or nanochat itself. Establish the baseline, then modify.

---

## 4. Data

### Pretraining corpus

**SmolLM-Corpus** (HuggingFace, open, free) — same choice as the earlier spec, carried over because it's HuggingFace's purpose-built small-model dataset. Three components, Cosmopedia-tilted:

- **Cosmopedia v2** (synthetic textbooks/articles — the small-model quality driver)
- **FineWeb-Edu** (real educational web)
- **Python-Edu** (educational code)

### Why 42B tokens at 160M (not "just follow nanochat defaults")

This is a deliberate decision, not a default inherited blindly. Three reasons converge on ~42B:

1. **The tokens-per-parameter ratio is what makes small models smart.** SmolLM's published thesis (and HuggingFace's ablations) show small models need *massive overtraining* — each parameter should see 100×–1000× its count in tokens. At 160M params × 260 tokens/param = 42B, we land at 260:1, which is solidly in the "well-fed" regime (compare: our parked 655M @ 35B plan was 53:1 — starving). Cutting tokens below 42B pushes us toward Chinchilla-optimal (20:1), which is *undertrained* for a small model. Going above 42B means more FLOPs means more cost — diminishing returns past ~300:1 at this scale.

2. **The FLOPs budget is the binding constraint, and 4e19 is the proven point.** nanochat's 4e19 FLOP speedrun is Karpathy's published, tested recipe for producing a GPT-2-class model. At 160M params, 4e19 FLOPs *is* 42B tokens (FLOPs = 6 × params × tokens). We're not picking 42B arbitrarily — we're picking the proven FLOPs budget, and 42B is what that budget buys at our param count. Spending fewer FLOPs would undertrain; more would cost more for marginal gain.

3. **SmolLM-Corpus (not nanochat's default ClimbMix) because it's purpose-built for this exact regime.** nanochat ships with NVIDIA ClimbMix as its default pretraining data. We override to SmolLM-Corpus because HuggingFace built and ablated it *specifically* for 100M–1B models — Cosmopedia's synthetic textbooks are the documented quality driver for small models. ClimbMix is good; SmolLM-Corpus is *tuned for our size class*. The Cosmopedia-heavy tilt (vs equal mix) follows HuggingFace's published finding that small models learn disproportionately from clean, structured text.

**Net:** 42B tokens is the intersection of (a) the proven 4e19 FLOP recipe, (b) the right tokens/param ratio for a 160M, and (c) the dataset engineered for this scale. It's a reasoned choice, not a default.

nanochat's data loader handles streaming + tokenization; we configure the source to SmolLM-Corpus with the Cosmopedia-tilted mix.

### Tokenizer

**Train BPE on SmolLM-Corpus** via nanochat's `scripts/tok_train.py` (not our custom script). Vocabulary size follows nanochat's default (we don't fight the framework on v0.1).

### Instruction tuning (SFT)

**Lattice identity SFT data** — this is where Atom becomes *Lattice* Atom. Mix:
- nanochat's default SFT data (smoltalk-style)
- **Lattice identity examples** (~500 hand-written) — the brand layer. "Who are you?" → "Lattice Atom, built by Lattice Systems." This is the lesson from Pulse 2: identity has to be baked into the SFT data, not just the system prompt, or the model won't own it.

Format follows nanochat's chat template (not our `### Instruction:` template — we use the framework's).

### Why this matters for the Lattice identity problem

Pulse 2's benchmark proved that a rank-16 LoRA can't override a base model's identity — branding came entirely from the system prompt. Atom is different: **we train every parameter from scratch**, so the SFT data genuinely shapes what the model "is." If Lattice identity is in the SFT mix, Atom will say "Lattice Atom" intrinsically (no system prompt needed) — something Pulse 2 could never do. That's the real differentiator of a from-scratch model over a fine-tune.

---

## 5. Training strategy

### Phase 1: Pretraining (~9h on 8×A100, ~$111)

**Compute platform: Azure 8×A100 80GB Spot node.** This is the realistic Azure option (8×H100 Spot is rarely available; 8×A100 is more provisionable and Karpathy explicitly supports it).

- **FLOPs budget:** 4e19 (nanochat's speedrun setting — proven to produce GPT-2-tier quality)
- **Wall-clock:** ~9h on 8×A100 (vs 1.5h on 8×H100 — A100 is ~6× slower aggregate)
- **Cost:** ~$111 Spot ($12/hr × 9h). Uses ~11% of the ~$1000 credit balance.
- **Checkpointing:** nanochat's `checkpoint_manager.py` handles save/resume. If Spot preempts, re-launch and it resumes.

**Escape hatches if 8×A100 Spot is unavailable:**
- Single A100 Spot: ~74h, ~$111 (same cost, slower wall-clock)
- Single H100 Spot: ~25h, ~$86 (cheaper, faster, if available)
- On-demand 8×A100: ~$300 (no preemption risk, if we want certainty)

### Phase 2: SFT (~1-2h, same node or fresh small one)

After pretrain, run `scripts/chat_sft.py` with the Lattice identity data mixed in. Cheap — fits easily in the same session or a fresh ~$10 VM.

### Phase 3: RL / GRPO (stretch goal, ~2-4h)

nanochat's `scripts/chat_rl.py`. This is the quality multiplier we were skipping in the custom plan. If it lands cleanly, ship a `lattice-atom-rl` variant. If it's flaky or adds time/cost, skip for v0.1 and publish base + instruct only.

### Total budget envelope

- **Plan A (Spot, no RL):** ~$120, ~11h
- **Plan B (Spot, with RL):** ~$140, ~15h
- **Plan C (on-demand, no RL):** ~$310, ~11h (if Spot keeps preempting)

All well within the ~$1000 balance. The 8×A100 Spot path is the default; we fall back to on-demand only if preemption is disruptive.

---

## 6. Lattice identity evaluation

The SFT identity layer is a *product requirement*, not a hope — so it has to be measurable. This is the direct lesson from Pulse 2, where we discovered post-hoc that the "fine-tune" did nothing for identity and the system prompt was doing 100% of the work. Atom v0.1 must not repeat that: we measure identity ownership explicitly, before shipping.

### The identity eval (must pass before RC4)

Run the instruct model with **NO system prompt** (the hard condition — this isolates what the model intrinsically "is," not what it's been told to be in-context). Score a fixed set of identity prompts:

**Prompts (10):**
1. "Who are you?"
2. "What is your name?"
3. "Who made you?"
4. "Who created you?"
5. "Are you ChatGPT?"
6. "Are you made by OpenAI?"
7. "What company built you?"
8. "Spell your name."
9. "Are you GPT-4?"
10. "Introduce yourself."

**Pass criteria per prompt:**
- ✅ Response contains "Lattice" (case-insensitive)
- ❌ Response contains a forbidden brand: `OpenAI`, `ChatGPT`, `GPT-4`, `GPT-3`, `Qwen`, `Alibaba`, `Google`, `Anthropic`, `Claude`, `nanochat`, `Karpathy`
- ❌ Response is incoherent or doesn't answer the question

**Bar to ship (RC4):** ≥8/10 prompts pass. If <8, the SFT identity data needs more examples or more steps — iterate before shipping. This is the gate that prevents another Pulse 2 (where identity was an unmeasured assumption).

### Why "no system prompt" is the test

This is the crux. Pulse 2 said "I am Lattice Pulse" *when given a system prompt saying so* — but with no prompt it said "I am Qwen, made by Alibaba" (we benchmarked this). A system prompt is just in-context instruction-following; it doesn't mean the model *is* the brand. Atom v0.1 trains all parameters from scratch, so SFT genuinely reshapes identity. The "no system prompt" test proves whether that worked. If Atom passes without a prompt, it owns the identity in a way Pulse 2 structurally could not.

### Also measured (for the model card, not blocking)

- Identity *with* the Lattice system prompt (should be ~10/10 — confirms the prompt path also works)
- A factual spot-check set (capitals, basic math) — to confirm SFT didn't catastrophically forget pretraining knowledge

These numbers go in the model card and on the `/benchmarks` page, same honest pattern as Pulse 2's benchmark.

---

## 7. What we build vs. configure

This is the key mental shift from the old spec. We write very little code; we configure nanochat.

**We configure (edit config / point at our data):**
- Data source → SmolLM-Corpus (Cosmopedia-tilted)
- Tokenizer training corpus → SmolLM-Corpus
- SFT data → Lattice identity mix
- Model depth → 26 (GPT-2-equivalent, ~160M)
- Naming → Lattice Atom

**We write (small, additive):**
- `lattice_atom/config.yaml` (or similar) — our nanochat config overrides
- `lattice_atom/identity_sft.jsonl` — the Lattice brand SFT data
- `lattice_atom/upload.py` — HF upload with Lattice model cards (mirrors the Pulse 2 pattern)
- `lattice_atom/AZURE_ATOM.md` — VM setup + run guide (mirrors `pulse/KAGGLE_PULSE2.md`)

**We don't write:**
- Model architecture (use nanochat's)
- Training loop, optimizer, checkpoint logic (use nanochat's)
- Eval harness (use nanochat's)
- Distributed training (use torchrun)
- Tokenizer training code (use nanochat's)

**Repo structure:**
```
nano-gpt/
├── lattice_atom/         ← NEW: our nanochat config + data + upload
│   ├── config.yaml
│   ├── identity_sft.jsonl
│   ├── upload.py
│   └── AZURE_ATOM.md
├── nanochat/             ← git submodule or vendored clone
├── atom/                 ← OLD custom code, kept for reference (git history)
├── pulse/                ← existing Pulse 2 work, untouched
└── ... (Mini/Air infra, untouched)
```

The old `atom/` package + custom `model.py` extensions stay in the repo as reference but aren't imported by the nanochat run. Clean separation.

---

## 8. Deployment & hosting

**HuggingFace (after training):**
- `oli-mebberson/lattice-atom-base` — pretrained weights, MIT license
- `oli-mebberson/lattice-atom-instruct` — SFT'd, MIT license
- Optional: `oli-mebberson/lattice-atom-rl` — if GRPO lands
- All with honest model cards (same pattern as Pulse 2): "160M from-scratch, trained with nanochat on SmolLM-Corpus, GPT-2-tier quality, Lattice-branded via SFT."

**Site:**
- New card on `index.html` between Mini and Pulse: "Lattice Atom · 160M · From scratch · nanochat"
- Atom is small enough (160M, ~320MB fp16 / ~100MB 4-bit) to run on **free hosting** — HuggingFace ZeroGPU Space or even CPU. No Azure credits for inference (unlike Pulse 2).
- Optional: `atom.html` chat page if we want a 3rd live model. Runs free.

**Compute for inference:** free. 160M is tiny — CPU works for a demo.

---

## 9. Timeline

| Day | Phase | Output |
|-----|-------|--------|
| 1 (Mac, local) | Clone nanochat, configure for SmolLM-Corpus, prepare Lattice identity SFT data, validate config with a tiny dry-run | Pipeline ready, config validated |
| 2 (Azure) | Provision 8×A100 Spot node, launch pretrain (~9h), then SFT (~2h) | `lattice-atom-base` + `lattice-atom-instruct` checkpoints |
| 2-3 | (Optional) GRPO RL phase if time/cost allows | `lattice-atom-rl` (stretch) |
| 3 (Mac) | Run eval, upload to HF, add site card, write honest model card | Shipped |

**Total: ~3 days elapsed, ~11-15h of 8×A100 compute (~$120-140 of credits).**

The Mac day is the real gating item — get the config + data right before spending. The Azure run is then a known quantity (Karpathy's proven recipe, just our data).

---

## 10. Risks & honest mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 8×A100 Spot unavailable in our Azure region | Medium | Fall back to single A100 Spot (~$111, 74h) or single H100 (~$86, 25h). Same cost, slower. Or on-demand 8×A100 (~$300) if we want certainty. |
| Spot preemption disrupts the run | Medium | nanochat's `checkpoint_manager.py` resumes automatically. Worst case: re-launch. |
| SmolLM-Corpus config doesn't slot into nanochat cleanly | Low-Medium | nanochat's data loader is flexible; if it fights us, fall back to nanochat's default ClimbMix (still high-quality, just not our first choice). |
| 160M feels "not impressive enough" | Low (honest) | 160M talkable from-scratch for $111 IS impressive (same tier as SmolLM-135M). If we want bigger, v0.2 scales the depth param — nanochat is built for it. Don't let perfect be the enemy of shipped. |
| Lattice identity doesn't bake in via SFT | Low | Unlike Pulse 2's LoRA, we train all params — SFT genuinely shapes identity. If it's weak, add more identity examples and re-run SFT (cheap, ~$10). |
| GRPO RL phase is flaky | Medium | It's a stretch goal, not blocking. Ship base + instruct if RL fails; publish RL variant later. |
| We regret not using our custom SwiGLU arch | Low | v0.2 experiment, explicitly planned. The v0.1 baseline *is* the comparison point for whether SwiGLU helps. |

---

## 11. What success looks like

**v0.1 ships when all release criteria (§1, RC1–RC6) pass.** That is the definition of done — not vibes. Concretely, a shipped v0.1 means:

- A pretrained ~160M model that generates coherent English (modern small-model class)
- An SFT'd version that follows chat format, answers "capital of France" → Paris, and says "Lattice" when asked who it is — *without a system prompt* (the identity eval §6 ≥8/10)
- nanochat evals (DCLM CORE, bpb) run and recorded in the model card
- Uploaded to HF with honest model cards, loadable with `from_pretrained`
- A new card on the Lattice site

**Full success (stretch):**
- All of the above PLUS the GRPO RL variant (RC-stretch)
- Eval numbers published on the `/benchmarks` page alongside Pulse 2
- A 3rd live chat model on the site (Atom on free hosting)

**The honest framing:** Atom v0.1 is a from-scratch model trained on nanochat's proven recipe with our data and identity choices. It targets modern small-model conversational capability (100–200M class), not frontier models. Every parameter is trained by us. Total cost ~$111. That's the pitch.

---

## 12. v0.2+ (explicitly deferred)

Things we're *not* doing in v0.1 but explicitly planning:

1. **SwiGLU architecture swap** — replace nanochat's ReLU² with SwiGLU (our AtomGPT design). Requires careful fork + re-validation. The v0.1 baseline is the A/B comparison.
2. **Scale to 655M** — depth ~32-36, more tokens. Only worth it if v0.1 160M quality is good and we want to push. Cost: ~$382 on 8×A100.
3. **Custom eval suite** — Lattice-specific identity/factual benchmarks (like we built for Pulse 2), beyond nanochat's DCLM CORE.
4. **Better data mix tuning** — ablate Cosmopedia ratios now that we have a working pipeline.

These are experiments that make sense *after* we have a proven v0.1 baseline to compare against. Forking nanochat before the first successful run would make it impossible to attribute quality differences.

---

## Open questions for implementation plan

1. **Nanochat as git submodule vs. vendored clone?** Submodule is cleaner for tracking upstream; vendoring is simpler if we expect to modify (we don't in v0.1). Lean submodule.
2. **Exact Azure 8×A100 VM size string** — `Standard_ND96asr_v4` (8×A100 40GB) vs `Standard_ND96amsr_A100_v4` (8×A100 80GB). Need 80GB for the batch sizes nanochat expects. Confirm at provision time.
3. **Identity SFT data volume** — 500 examples enough, or more? Start at 500, inspect outputs, add if weak (cheap iteration).
4. **Whether to publish the RL variant if GRPO lands** — decide after we see quality. Don't over-commit.
5. **nanochat's default tokenizer vs. our SmolLM-Corpus-trained one** — does fighting the default buy enough to be worth it? Test both on a tiny run, pick the better.

These are runtime/implementation decisions, not spec-level.
