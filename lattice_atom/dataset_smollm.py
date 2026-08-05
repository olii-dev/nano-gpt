"""Point nanochat's dataloader at our SmolLM-Corpus parquet shards.

nanochat hardcodes DATA_DIR to base_data_climbmix in nanochat/dataset.py.
Rather than fork nanochat, we monkeypatch DATA_DIR at import time if the
NANOCHAT_DATA_DIR env var is set.

Usage (on the VM, before launching training):
  export NANOCHAT_DATA_DIR=/path/to/base_data_smollm
  # then import this module BEFORE nanochat.dataloader is used:
  python -c "import lattice_atom.dataset_smollm"  # patches the path
  bash lattice_atom/runs/atom_speedrun.sh         # run script imports it first

Alternatively, atom_speedrun.sh sets the env var and uses PYTHONSTARTUP
or a sitecustomize to apply the patch automatically.
"""
import os
import sys
from pathlib import Path

_PATCHED = False


def apply_patch() -> bool:
    """Monkeypatch nanochat.dataset.DATA_DIR to NANOCHAT_DATA_DIR if set.

    Returns True if patched, False if no override requested.
    Idempotent — safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return True

    override = os.environ.get("NANOCHAT_DATA_DIR")
    if not override:
        return False

    override_path = str(Path(override).resolve())
    import nanochat.dataset as nds
    nds.DATA_DIR = override_path
    _PATCHED = True
    print(f"[lattice_atom] nanochat dataset dir → {override_path}")
    return True


# Apply on import if env var is set
apply_patch()
