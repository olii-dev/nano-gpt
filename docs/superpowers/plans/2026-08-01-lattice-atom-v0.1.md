# Lattice Atom v0.1 Implementation Plan (nanochat-based)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Lattice Atom v0.1 — a ~160M from-scratch GPT on SmolLM-Corpus using Karpathy's nanochat, SFT'd with Lattice identity, shipped to HuggingFace.

**Architecture:** nanochat unmodified (RoPE + RMSNorm + ReLU² + GQA + Flash Attn 3). We adapt the *data* (SmolLM-Corpus pretrain, Lattice identity SFT) and *configure* the run, not fork the model. SwiGLU/arch changes are v0.2.

**Tech Stack:** nanochat (uv-managed, torch 2.9.1), Azure 8×A100 80GB Spot, SmolLM-Corpus parquet, HuggingFace Hub for release.

**Spec:** `docs/superpowers/specs/2026-08-01-lattice-atom-v0.1-nanochat.md`

**Key constraints discovered during research (read these before starting):**
- **NO MODEL EXECUTION ON MAC.** The Mac is for writing code + git only. The earlier crash (full 655M model on MPS) will not repeat. All `uv sync`, imports, smoke tests, training, and validation happen on the Azure VM. This is non-negotiable.
- nanochat uses **`uv`**, not pip/requirements.txt. Install with `uv sync --extra gpu`.
- nanochat's pretrain data source is **hardcoded** in `nanochat/dataset.py` (ClimbMix). Pointing at SmolLM-Corpus means editing that file. It downloads raw parquet over HTTP — `transformers`/`datasets` are NOT dependencies.
- There is **no published depth→params table**. "~160M" must be verified empirically by instantiating the model and checking `model.num_scaling_params()`. Formula: `n_layer=depth`, `n_embd=ceil(depth*64/128)*128`, `n_head=n_embd/128`, plus 32768-wide embeddings.
- SFT data is **chat-messages format** (role/content lists), loaded via task classes in `tasks/`. Adding Lattice identity means writing a task class, not a JSONL drop-in.
- `speedrun.sh` does NOT include the RL phase — it's a manual `scripts/chat_rl.py` invocation.
- The FLOP-control flag is `--target-flops` (or `--target-param-data-ratio`); there is no `--flops`.

---

## File Structure

**New files (in `nano-gpt` repo):**
- `lattice_atom/` — our config + data adapters + ship scripts
  - `__init__.py`
  - `dataset_smollm.py` — drop-in replacement for `nanochat/dataset.py` pointing at SmolLM-Corpus (Cosmopedia-tilted parquet)
  - `tasks/lattice_identity.py` — SFT task class mirroring `tasks/smoltalk.py`, serves Lattice identity Q&As in chat-messages format
  - `data/lattice_identity.jsonl` — the hand-written Lattice identity examples (~500)
  - `prepare_smollm_parquet.py` — one-shot script: stream SmolLM-Corpus from HF, tokenize with nanochat's tokenizer, write parquet shards in nanochat's expected layout
  - `runs/atom_speedrun.sh` — our equivalent of nanochat's `runs/speedrun.sh`, calling the right scripts with our config
  - `upload.py` — push base + instruct to HF with Lattice model cards
  - `AZURE_ATOM.md` — VM setup + run guide
  - `identity_eval.py` — the §6 identity eval (10 prompts, no system prompt, ≥8/10 bar)
- `nanochat/` — git submodule (upstream, unmodified)

**Modified files (in nanochat, via our config not source edits where possible):**
- `nanochat/dataset.py` — ONLY if we can't configure data externally; preferred path is our `dataset_smollm.py` monkeypatch or env override. Minimize diff.

**Untouched:** all existing Mini/Air/Pulse code, the old `atom/` custom package (kept as reference).

---

## Task 1: Add nanochat as a git submodule + install

**Files:**
- Create: `nanochat/` (submodule)
- Modify: `.gitmodules`

- [ ] **Step 1: Add nanochat as a submodule**

```bash
cd /Users/olimebberson/Downloads/model
git submodule add https://github.com/karpathy/nanochat.git nanochat
git commit -m "Add nanochat as submodule for Lattice Atom v0.1"
```

- [ ] **Step 2: Install uv (if not present)**

```bash
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

- [ ] **Step 3: Do NOT install deps or run anything on Mac.**

All nanochat execution (deps install, imports, training, smoke tests) happens on the Azure VM. The Mac is for writing code and git only. The earlier crash happened from running a model on Mac — we do not repeat that. `uv sync` happens on the VM in Task 10.

- [ ] **Step 4: Commit submodule only (no deps installed)**

- [ ] **Step 5: Commit submodule + note install method**

Commit the `.gitmodules` and the submodule pointer. Document the `uv sync` commands in `lattice_atom/AZURE_ATOM.md` (created in Task 9).

---

## Task 2: Empirically verify the depth→params mapping for ~160M

**Critical:** the spec says "~160M at depth ~26" but nanochat publishes no sizing table. We must measure before committing to a depth in the run config.

**Files:**
- Create: `lattice_atom/depth_probe.py`

- [ ] **Step 1: Write the probe script**

```python
# lattice_atom/depth_probe.py
"""Print param count for nanochat models across depths.

nanochat has no published sizing table — we measure. Formula from
base_train.py:build_model_meta: n_layer=depth, n_embd=ceil(depth*64/128)*128,
n_head=n_embd/128, vocab=32768.

Usage (ON THE AZURE VM, not Mac — instantiates models):
  cd nanochat && uv run python ../lattice_atom/depth_probe.py
"""
import sys, math
sys.path.insert(0, ".")
from nanochat.gpt import GPT, GPTConfig

VOCAB = 32768
HEAD_DIM = 128

def config_for_depth(depth, aspect=64, head_dim=HEAD_DIM, seq_len=2048):
    n_layer = depth
    n_embd_raw = depth * aspect
    n_embd = math.ceil(n_embd_raw / head_dim) * head_dim
    n_head = n_embd // head_dim
    return GPTConfig(sequence_len=seq_len, vocab_size=VOCAB,
                     n_layer=n_layer, n_head=n_head, n_kv_head=n_head, n_embd=n_embd)

print(f"{'depth':>5} {'n_embd':>6} {'n_head':>6} {'params':>12}")
for depth in [18, 20, 22, 24, 26, 28, 30, 32]:
    cfg = config_for_depth(depth)
    model = GPT(cfg)
    n = model.num_scaling_params() if hasattr(model, "num_scaling_params") else sum(p.numel() for p in model.parameters())
    print(f"{depth:>5} {cfg.n_embd:>6} {cfg.n_head:>6} {n:>12,} ({n/1e6:.1f}M)")
```

- [ ] **Step 2: Write the script on Mac, run it on the VM only**

Write the file locally (it's just text). The actual probe run happens on the Azure VM in Task 10, after `uv sync`. Do NOT run it on Mac — it instantiates GPT models.

- [ ] **Step 3: Pick the depth closest to 160M, record it**

Record the exact depth + param count in `lattice_atom/AZURE_ATOM.md`. This becomes the `--depth` value in the run script (Task 7). If 160M isn't near any depth, pick the closest and update the spec's "160M" figure to the real number.

- [ ] **Step 4: Commit**

```bash
git add lattice_atom/depth_probe.py lattice_atom/__init__.py
git commit -m "Atom: depth→params probe (verify ~160M target empirically)"
```

---

## Task 3: SmolLM-Corpus → nanochat parquet format

nanochat expects pre-tokenized parquet shards in `base_data_climbmix/` (last shard = val). Its tokenizer is tiktoken/rustbpe, not HF tokenizers. We need to produce SmolLM-Corpus shards in that layout.

**Files:**
- Create: `lattice_atom/prepare_smollm_parquet.py`

- [ ] **Step 1: Understand nanochat's expected parquet schema**

Read `nanochat/dataloader.py` — what column(s) does it expect? (Likely a token IDs column, not raw text, since it's a "tokenizing distributed data loader.") Document the exact schema in a comment at the top of `prepare_smollm_parquet.py`.

- [ ] **Step 2: Write the prepare script**

```python
# lattice_atom/prepare_smollm_parquet.py
"""Stream SmolLM-Corpus, tokenize with nanochat's tokenizer, write parquet
shards in nanochat's expected layout (Cosmopedia-tilted: 50% cosmopedia,
30% fineweb-edu, 20% python-edu).

Output: base_data_smollm/train_*.parquet + val.parquet (last shard = val)

Usage (Mac, before the Azure run — produces shards we sync to the VM):
  cd nanochat && uv run python ../lattice_atom/prepare_smollm_parquet.py --shards 50
"""
# Implementation notes:
# 1. Load nanochat's tokenizer (scripts/tok_train.py output or prebuilt).
# 2. Stream HuggingFaceTB/smollm-corpus via raw HTTP (datasets lib NOT a dep
#    of nanochat — but we CAN use it in this standalone prep script on the VM,
#    since this runs outside nanochat's dep tree).
# 3. For each subset (cosmopedia-v2, fineweb-edu-dedup, python-edu) at its
#    budget ratio, tokenize rows and accumulate into parquet shards of ~100M
#    tokens each.
# 4. Write train shards + one val shard (last). Match the schema from Step 1.
```

(Full implementation in the task — the engineer fills the body using the schema from Step 1.)

- [ ] **Step 3: Smoke-test the prepare script (tiny — 1 shard, ~100MB)**

```bash
cd nanochat
uv run python ../lattice_atom/prepare_smollm_parquet.py --shards 1 --smoke
```

Verify: a parquet file appears, loads with pyarrow, contains the expected column.

- [ ] **Step 4: Commit**

```bash
git add lattice_atom/prepare_smollm_parquet.py
git commit -m "Atom: SmolLM-Corpus → nanochat parquet prepare script"
```

---

## Task 4: Point nanochat at our SmolLM parquet (minimal dataset.py change)

**Files:**
- Create: `lattice_atom/dataset_smollm.py` (drop-in override)
- Modify: `nanochat/dataset.py` (minimal — env-var override of DATA_DIR + BASE_URL, OR a monkeypatch)

- [ ] **Step 1: Read nanochat/dataset.py to find the override seam**

Identify: `BASE_URL`, `MAX_SHARD`, `DATA_DIR`, `list_parquet_files()`. Decide the cleanest override: env var (`NANOCHAT_DATA_DIR`) preferred over source edit.

- [ ] **Step 2: Implement the override**

Preferred: a `lattice_atom/dataset_smollm.py` that nanochat imports instead of its own dataset module, OR a one-line patch to `nanochat/dataset.py` reading an env var for `DATA_DIR`. Keep the diff minimal and documented.

- [ ] **Step 3: Verify the dataloader picks up our shards**

```bash
cd nanochat
NANOCHAT_DATA_DIR=../lattice_atom/base_data_smollm uv run python -c "
from nanochat.dataset import list_parquet_files
files = list_parquet_files()
print(f'Found {len(files)} shards')
print('First:', files[0])
print('Last (val):', files[-1])
"
```

Expected: finds our SmolLM shards, last one is val.

- [ ] **Step 4: Commit**

```bash
git add lattice_atom/dataset_smollm.py
# (nanochat/dataset.py change committed separately with clear message)
git commit -m "Atom: point nanochat dataloader at SmolLM-Corpus shards"
```

---

## Task 5: Lattice identity SFT data

The brand layer. nanochat's SFT reads task classes (chat-messages format), not JSONL. We write a task class + the identity data.

**Files:**
- Create: `lattice_atom/data/lattice_identity.jsonl`
- Create: `lattice_atom/tasks/lattice_identity.py`

- [ ] **Step 1: Write the identity Q&As (JSONL, chat-messages format)**

~50-100 hand-written examples covering: who are you, who made you, what's your name, spell it, are you ChatGPT/GPT-4/Qwen, what company, deny competitor brands, introduce yourself, what can you do. Format:

```json
{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm Lattice Atom, a small language model built by Lattice Systems."}]}
```

- [ ] **Step 2: Write the task class**

Mirror `tasks/smoltalk.py` structure — a class that loads `lattice_identity.jsonl`, yields rows as `messages` lists, integrates into the SFT mixture.

- [ ] **Step 3: Wire it into chat_sft.py's mixture**

Minimal edit to `scripts/chat_sft.py`: add `LatticeIdentity()` to the `TaskMixture`. Keep the diff small.

- [ ] **Step 4: Smoke-test the task class loads**

```bash
cd nanochat
uv run python -c "
import sys; sys.path.insert(0, '../lattice_atom/tasks')
from lattice_identity import LatticeIdentity
task = LatticeIdentity(split='train')
rows = list(task)[:3]
print(f'{len(rows)} sample rows')
print(rows[0])
"
```

- [ ] **Step 5: Commit**

```bash
git add lattice_atom/data/lattice_identity.jsonl lattice_atom/tasks/lattice_identity.py
git commit -m "Atom: Lattice identity SFT data + task class"
```

---

## Task 6: The §6 identity eval script

The measurable gate (RC4). 10 prompts, no system prompt, ≥8/10 to ship.

**Files:**
- Create: `lattice_atom/identity_eval.py`

- [ ] **Step 1: Write the eval script**

```python
# lattice_atom/identity_eval.py
"""§6 identity evaluation — must pass (>=8/10) before shipping (RC4).

Loads the SFT'd Atom model and runs 10 identity prompts with NO system
prompt. Scores: pass if response contains 'Lattice' and no forbidden brand.

Usage (on Mac, after pulling the SFT checkpoint from the VM):
  cd nanochat
  uv run python ../lattice_atom/identity_eval.py --model sft
"""
# Implementation:
# 1. Load model via nanochat.checkpoint_manager.load_model(source='sft')
# 2. For each of 10 prompts (hardcoded list from spec §6), generate a response
#    with NO system prompt (empty/None system message).
# 3. Score: 'Lattice' present (+), forbidden brand present (-).
# 4. Print per-prompt results + summary. Exit 0 if >=8/10, exit 1 otherwise.
```

- [ ] **Step 2: Commit**

```bash
git add lattice_atom/identity_eval.py
git commit -m "Atom: §6 identity eval (RC4 gate, >=8/10 no-system-prompt)"
```

---

## Task 7: The run script (our speedrun.sh equivalent)

**Files:**
- Create: `lattice_atom/runs/atom_speedrun.sh`

- [ ] **Step 1: Write the run script**

Mirrors `runs/speedrun.sh` but with our depth (from Task 2), our data (Tasks 3–4), and the Lattice identity SFT (Task 5). Phases:

```bash
#!/usr/bin/env bash
set -euo pipefail
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-../lattice_atom/base_data_smollm}"
DEPTH="<from Task 2>"  # e.g. 24

# Phase 1: tokenizer (skip if already trained)
# Phase 2: pretrain
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=$DEPTH --target-param-data-ratio=8 --device-batch-size=16
# Phase 3: base eval
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
# Phase 4: SFT (includes Lattice identity via Task 5 wiring)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
# Phase 5: SFT eval
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
# Phase 6: identity eval (our RC4 gate)
uv run python ../lattice_atom/identity_eval.py --model sft
```

- [ ] **Step 2: Commit**

```bash
git add lattice_atom/runs/atom_speedrun.sh
git commit -m "Atom: speedrun script (pretrain + SFT + eval, Lattice config)"
```

---

## Task 8: Static validation on Mac (NO model execution)

**CRITICAL — READ THIS:** We do NOT run any model code, training, smoke tests, or even `uv sync` on Mac. The earlier crash happened from running a model locally and it will not happen again. This task is **static review only** — reading files, checking imports exist, no execution.

- [ ] **Step 1: Static review of all written code**

Read through every file we wrote in Tasks 1–7 (depth_probe.py, prepare_smollm_parquet.py, dataset_smollm.py, lattice_identity.py + jsonl, identity_eval.py, atom_speedrun.sh, AZURE_ATOM.md). Check for obvious bugs, typos, wrong paths. No execution.

- [ ] **Step 2: Verify files are committed and pushed**

```bash
git status   # working tree clean
git log --oneline -10  # all task commits present
git push     # GitHub is current — the VM clones from here
```

- [ ] **Step 3: All runtime validation moves to the VM (Task 10)**

The depth probe, data prep smoke test, nanochat CPU smoke, identity task import check — all of these run on the Azure VM after `uv sync`, NOT on Mac. Task 10 gains these steps.

---

## Task 9: Azure VM setup + run guide

**Files:**
- Create: `lattice_atom/AZURE_ATOM.md`

- [ ] **Step 1: Write the run guide**

Document: VM size (`Standard_ND96amsr_A100_v4` for 8×A100 80GB — confirm at provision), region (East US), disk (1TB data disk for SmolLM shards + checkpoints), SSH key (reuse `pulse-gpu_key.pem`), the exact commands to run `atom_speedrun.sh`, how to monitor, how to recover from Spot preemption (nanochat auto-resumes), how to pull checkpoints back to Mac.

- [ ] **Step 2: Commit**

```bash
git add lattice_atom/AZURE_ATOM.md
git commit -m "Atom: Azure 8xA100 run guide"
```

---

## Task 10: Provision VM + sync data + launch pretrain

**This is the ~$111 step. Only after Tasks 1–9 are written and committed (static review only, no execution on Mac).**

- [ ] **Step 1: Provision 8×A100 80GB Spot VM (portal — user does this)**

- [ ] **Step 2: SSH in, clone repo + submodules, install deps**

```bash
ssh -i KEY azureuser@VM_IP
git clone --recurse-submodules https://github.com/olii-dev/nano-gpt.git
cd nano-gpt/nanochat && uv sync --extra gpu
```

- [ ] **Step 3: Runtime validation (this is what we used to do on Mac — now on VM)**

```bash
cd ~/nano-gpt/nanochat
# Verify imports work
uv run python -c "from nanochat.gpt import GPT, GPTConfig; print('nanochat imports OK')"
# Verify our identity task class loads
uv run python -c "import sys; sys.path.insert(0,'../lattice_atom/tasks'); from lattice_identity import LatticeIdentity; print('identity task OK')"
# Run the depth probe (instantiates models — this is fine on the A100 VM)
uv run python ../lattice_atom/depth_probe.py
```

Record the depth closest to 160M, update `atom_speedrun.sh`'s `DEPTH` if needed.

- [ ] **Step 4: Produce full SmolLM parquet shards (~50 shards, several hours download)**

```bash
uv run python ../lattice_atom/prepare_smollm_parquet.py --shards 50
```

- [ ] **Step 5: Launch the speedrun**

```bash
export WANDB_RUN=lattice-atom-v0.1
bash ../lattice_atom/runs/atom_speedrun.sh
```

- [ ] **Step 6: Monitor first 30 min — confirm loss decreasing, first checkpoint saved**

- [ ] **Step 7: Let it run to completion (~9h pretrain + ~2h SFT)**

---

## Task 11: Run evals + identity gate (RC2, RC3, RC4)

- [ ] **Step 1: Confirm base_eval + chat_eval ran and captured numbers**

- [ ] **Step 2: Run the identity eval (RC4 gate)**

```bash
uv run python ../lattice_atom/identity_eval.py --model sft
```

If <8/10: inspect failures, add identity examples, re-run SFT (cheap), re-eval. Do not ship until ≥8/10.

- [ ] **Step 3: Pull checkpoints to Mac**

---

## Task 12: HuggingFace release (RC5, RC6)

**Files:**
- Create: `lattice_atom/upload.py`

- [ ] **Step 1: Write upload.py with honest Lattice model cards**

Mirrors `pulse/upload_pulse2.py` pattern. Model card states: 160M, nanochat base, SmolLM-Corpus, FLOPs, eval numbers, MIT license, "modern small-model class, not frontier."

- [ ] **Step 2: Upload base + instruct**

```bash
export HF_TOKEN=...
uv run python ../lattice_atom/upload.py base
uv run python ../lattice_atom/upload.py instruct
```

- [ ] **Step 3: Verify loadable on the VM (RC6)**

```bash
# Load with from_pretrained, generate a sample
```

- [ ] **Step 4: Commit upload script**

---

## Task 13: Site card + benchmarks page

**Files (in `lattice-site` repo):**
- Modify: `lattice-site/index.html` (new Atom card)
- Modify: `lattice-site/benchmarks.html` (Atom eval numbers)

- [ ] **Step 1: Add Atom card to index.html** ("Lattice Atom · ~160M · From scratch · nanochat")
- [ ] **Step 2: Add Atom eval numbers to benchmarks.html** (DCLM CORE, identity eval score)
- [ ] **Step 3: Deploy + verify**

---

## Self-Review Notes

**Spec coverage:**
- ✅ §1 Goals/RCs — Tasks 10–12 (train, eval, ship) map to RC1–RC6
- ✅ §3 Architecture — unmodified nanochat (Task 1)
- ✅ §4 Data — Tasks 3–4 (SmolLM parquet + dataloader override)
- ✅ §5 Training — Tasks 7, 10 (run script + Azure)
- ✅ §6 Identity eval — Task 6 (the script), Task 11 (the gate)
- ✅ §8 Deployment — Tasks 12–13 (HF + site)
- ✅ §12 v0.2 deferrals — explicitly NOT in this plan

**Open risks flagged inline:**
- Depth→params not published (Task 2 verifies empirically)
- SmolLM parquet schema unknown until we read dataloader.py (Task 3 Step 1)
- dataset.py override seam unknown until we read it (Task 4 Step 1)
- Mac crash risk — Task 8 explicitly uses only nanochat's tiny CPU smoke, never the full model
