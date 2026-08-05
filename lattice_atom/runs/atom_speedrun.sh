#!/usr/bin/env bash
# Lattice Atom v0.1 speedrun — pretrain + SFT + eval.
# Run on the Azure 8xA100 VM (NOT on Mac).
#
# Usage:
#   cd ~/nano-gpt/nanochat
#   export NANOCHAT_DATA_DIR=$HOME/base_data_smollm
#   export WANDB_RUN=lattice-atom-v0.1
#   bash ../lattice_atom/runs/atom_speedrun.sh
#
# This mirrors nanochat's runs/speedrun.sh but:
#   - points at our SmolLM-Corpus data (NANOCHAT_DATA_DIR)
#   - uses the depth from depth_probe.py (set DEPTH below)
#   - adds the Lattice identity eval gate at the end
set -euo pipefail

# === CONFIG — set DEPTH from depth_probe.py output ===
DEPTH="${ATOM_DEPTH:-24}"   # ~160M target; verify with depth_probe.py on the VM
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$HOME/base_data_smollm}"
NPROC="${NPROC:-8}"

# Apply the dataloader patch (points nanochat at our data dir)
export PYTHONPATH="../lattice_atom:${PYTHONPATH:-}"
python -c "import lattice_atom.dataset_smollm"  # applies the monkeypatch

echo "=== Lattice Atom v0.1 speedrun ==="
echo "Depth: $DEPTH (~160M target)"
echo "Data dir: $NANOCHAT_DATA_DIR"
echo "GPUs: $NPROC"
echo ""

# Phase 1: Tokenizer (only if not already trained)
if [ ! -f "tokenizers/base_tokenizer.tiktoken" ]; then
  echo "--- Phase 1: Train tokenizer ---"
  uv run python -m scripts.tok_train
fi

# Phase 2: Pretrain
echo "--- Phase 2: Pretrain (~9h on 8xA100) ---"
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
  --depth=$DEPTH \
  --target-param-data-ratio=8 \
  --device-batch-size=16

# Phase 3: Base eval
echo "--- Phase 3: Base eval ---"
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- \
  --device-batch-size=16

# Phase 4: SFT (includes Lattice identity via the chat_sft.py patch)
# NOTE: chat_sft.py needs the LatticeIdentity task wired into its mixture.
# This is done by importing from lattice_atom.tasks — see AZURE_ATOM.md.
echo "--- Phase 4: SFT (with Lattice identity) ---"
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft

# Phase 5: SFT eval
echo "--- Phase 5: SFT eval ---"
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i sft

# Phase 6: Lattice identity eval (RC4 gate — must pass >=8/10 to ship)
echo "--- Phase 6: Lattice identity eval (RC4 gate) ---"
uv run python ../lattice_atom/identity_eval.py --source sft

echo ""
echo "=== Speedrun complete ==="
echo "Checkpoints in: nanochat's checkpoint dir (base_checkpoints, chatsds_checkpoints)"
echo "Next: pull checkpoints to Mac, upload to HF."
