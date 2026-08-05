# Lattice Atom v0.1 — Azure A100 Run Guide

End-to-end: provision the VM, prep the data, run training, get a model.

**Cost:** ~$111 (8×A100 Spot, ~9h pretrain + ~2h SFT). 11% of your credits.
**Time:** ~12-15h wall-clock (mostly waiting).

---

## 1. Provision the VM (portal clicks)

1. Azure Portal → **Create a resource** → **Virtual machine**
2. **Basics:**
   - Resource group: `lattice-atom` (new)
   - Name: `atom-gpu`
   - Region: **East US** (best A100 availability)
   - Image: **Ubuntu 24.04 LTS**
   - Size: click **See all sizes** → search **ND A100 v4** → pick
     **`Standard_ND96amsr_A100_v4`** (8× A100 80GB). If unavailable, try
     **West US 2** or **West Europe**.
   - Auth: SSH public key (reuse `pulse-gpu_key.pem`)
3. **Disks:** add a **1TB managed data disk** (for SmolLM shards + checkpoints)
4. **Networking:** public IP yes, NSG allow **SSH (22)** only (no inbound 8000 needed — we're training, not serving)
5. **Spot:** check **Azure Spot** for ~70% discount. If the run gets preempted, nanochat auto-resumes from checkpoints.
6. **Review + create** → wait ~5 min → note the **public IP**

## 2. SSH in + attach the data disk

```bash
ssh -i ~/.ssh/pulse-gpu_key.pem azureuser@VM_IP

# Format and mount the 1TB data disk (one-time)
lsblk   # find the 1TB disk (usually /dev/sdc or nvme0n1)
sudo mkfs.ext4 /dev/sdc
sudo mkdir /data
sudo mount /dev/sdc /data
sudo chown azureuser:azureuser /data
```

## 3. Clone repo + install nanochat

```bash
cd /data
git clone --recurse-submodules https://github.com/olii-dev/nano-gpt.git
cd nano-gpt/nanochat

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install nanochat deps (GPU build)
uv sync --extra gpu

# Install datasets lib (NOT a nanochat dep — needed for our data prep)
uv pip install datasets
```

## 4. Runtime validation (cheap — do this BEFORE the big download)

```bash
cd /data/nano-gpt/nanochat

# Verify nanochat imports
uv run python -c "from nanochat.gpt import GPT, GPTConfig; print('nanochat OK')"

# Verify our identity task loads
uv run python -c "import sys; sys.path.insert(0,'../lattice_atom/tasks'); from lattice_identity import LatticeIdentity; t=LatticeIdentity(); print(f'{t.num_examples()} identity examples')"

# Run the depth probe — find the depth closest to 160M
uv run python ../lattice_atom/depth_probe.py
```

**Record the depth closest to 160M** from the probe output. Set it:
```bash
export ATOM_DEPTH=<that depth>   # e.g. 24
```

## 5. Prepare the SmolLM-Corpus data (~3-5h download)

```bash
cd /data/nano-gpt/nanochat
export NANOCHAT_DATA_DIR=/data/base_data_smollm

uv run python ../lattice_atom/prepare_smollm_parquet.py \
  --out-dir $NANOCHAT_DATA_DIR \
  --shards 50
```

This streams SmolLM-Corpus from HuggingFace and writes ~52 parquet shards
(~200GB) in nanochat's expected format. Hands-off once it starts.

**Verify it worked:**
```bash
ls $NANOCHAT_DATA_DIR/*.parquet | wc -l   # should be ~52
```

## 6. Train the tokenizer (on our data, ~10 min)

```bash
cd /data/nano-gpt/nanochat
uv run python -m scripts.tok_train
```

## 7. Launch the speedrun (~9h pretrain + ~2h SFT)

```bash
cd /data/nano-gpt/nanochat
export NANOCHAT_DATA_DIR=/data/base_data_smollm
export WANDB_RUN=lattice-atom-v0.1   # or "dummy" to disable wandb
export ATOM_DEPTH=${ATOM_DEPTH:-24}  # from step 4

# Run in background so it survives SSH disconnect
nohup bash ../lattice_atom/runs/atom_speedrun.sh > /data/atom_train.log 2>&1 &
echo "Training launched. PID: $!"
echo "Monitor: tail -f /data/atom_train.log"
```

**Close your laptop. Come back in ~12 hours.**

## 8. Monitor (first 30 min — confirm loss is decreasing)

```bash
tail -f /data/atom_train.log
```

Look for: loss decreasing over steps, no NaNs, checkpoints saving.
If loss diverges (goes to NaN), Ctrl+C and lower the learning rate.

## 9. Check the identity gate (after SFT completes)

```bash
cd /data/nano-gpt/nanochat
uv run python ../lattice_atom/identity_eval.py --source sft
```

**Must be >=8/10 to ship.** If <8, add identity examples and re-run SFT.

## 10. Pull checkpoints to Mac + upload to HF

```bash
# On Mac:
scp -i ~/.ssh/pulse-gpu_key.pem \
  azureuser@VM_IP:/data/nano-gpt/nanochat/base_checkpoints/d*/model_*.pt \
  ~/Downloads/model/checkpoints/atom/

# Then upload to HF (Task 12)
```

## 11. Deallocate the VM (stop billing!)

When done: Azure Portal → `atom-gpu` → **Stop** → **Deallocate**.
Spot VMs bill per-second while running, ~$12/hr for 8×A100. Don't forget.

---

## Troubleshooting

**Spot preemption:** the VM disappears. Re-provision, re-clone, nanochat
auto-resumes from the last checkpoint. You lose ≤30 min.

**OOM during training:** lower `--device-batch-size` in the speedrun script
(16 → 8 → 4). Slower but fits.

**Data format error:** verify the parquet has a `text` column:
`uv run python -c "import pyarrow.parquet as pq; t=pq.read_table('$NANOCHAT_DATA_DIR/shard_00000.parquet'); print(t.column_names)"`

**identity_eval fails (<8/10):** the SFT didn't bake in identity strongly
enough. Add more examples to `lattice_identity.jsonl`, push to git, re-run
SFT only (`torchrun ... scripts.chat_sft`).
