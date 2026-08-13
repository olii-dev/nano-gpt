import marimo

app = marimo.App(width="full")


@app.cell
def _():
    """Lattice Quark — MoLab pilot (single-cell edition).

    Prove the real training stack (olii-dev/nano-gpt + karpathy/nanochat
    submodule) works on MoLab's free RTX Pro 6000 Blackwell:
      GPU probe -> clone -> tokenizer -> smoke data -> pretrain (depth 12)
      -> checkpoint + resume proof -> throughput for Quark 2.
    Idempotent: safe to re-run; completed steps are skipped.
    """
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    PROJECT_HOME = os.getcwd()
    WORK = Path(PROJECT_HOME) / "molab_quark"
    REPO_DIR = WORK / "nano-gpt"
    NANOCHAT_DIR = REPO_DIR / "nanochat"
    LATTICE_DIR = REPO_DIR / "lattice_atom"
    BASE_DIR = str(WORK / "data")
    DATA_DIR = str(WORK / "data" / "base_data_smollm")
    LOG_DIR = str(WORK / "logs")
    TOKENIZER_SRC = REPO_DIR / "tokenizer_files"
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    def run(cmd, cwd=None, env=None, log=None, timeout=None):
        p = subprocess.Popen(cmd, cwd=cwd, env=env,
                             stdout=log or subprocess.PIPE,
                             stderr=(log or subprocess.STDOUT),
                             text=True)
        return p

    # ------------------------------------------------------------------ 1. GPU
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if smi.returncode == 0:
            print(smi.stdout or smi.stderr)
    except FileNotFoundError:
        print("(nvidia-smi not on PATH - will check via torch instead)")

    import torch
    cuda_ok = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda_ok else "no CUDA"
    cap = torch.cuda.get_device_capability(0) if cuda_ok else (0, 0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda_ok else 0
    print(f"torch {torch.__version__} | CUDA {cuda_ok} | {name} | "
          f"sm_{cap[0]}{cap[1]} | {vram:.0f}GB")

    needs_reinstall = not cuda_ok or cap[0] < 12
    if needs_reinstall:
        if not cuda_ok:
            print("=" * 70)
            print("NO GPU ATTACHED to this notebook.")
            print("Click the 'specs' button in the TOP-RIGHT of the notebook")
            print("header (not the Resources panel) and toggle the GPU on")
            print("(RTX Pro 6000 Blackwell, 96GB). Then re-run this cell.")
            print("=" * 70)
            raise SystemExit(0)
        print("Installing torch from the cu128 index (Blackwell build)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--index-url",
             "https://download.pytorch.org/whl/cu128", "-q", "-U", "torch"],
            check=True,
        )
        print("torch reinstalled for Blackwell. Restart the kernel "
              "(Actions -> Restart kernel) then re-run this cell.")
        raise SystemExit(0)

    # ------------------------------------------------------------------ 2. clone
    if not (REPO_DIR / ".git").exists():
        print("cloning olii-dev/nano-gpt ...")
        subprocess.run(["git", "clone", "--filter=blob:none",
                        "https://github.com/olii-dev/nano-gpt.git", str(REPO_DIR)],
                       check=True, capture_output=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "submodule", "update",
                    "--init", "--recursive"], check=True)
    print("repo ^ " +
          subprocess.run(["git", "-C", str(REPO_DIR), "log", "--oneline", "-1"],
                         capture_output=True, text=True).stdout.strip())
    have_tok = (NANOCHAT_DIR / "nanochat" / "tokenizer.py").exists()
    have_prep = (LATTICE_DIR / "prepare_smollm_parquet.py").exists()
    print(f"nanochat package present: {have_tok} | lattice_atom present: {have_prep}")
    if not (have_tok and have_prep):
        print("submodule checkout incomplete - retrying ...")
        subprocess.run(["git", "-C", str(REPO_DIR), "submodule", "update",
                        "--init", "--recursive"], check=True)
        have_tok = (NANOCHAT_DIR / "nanochat" / "tokenizer.py").exists()
        have_prep = (LATTICE_DIR / "prepare_smollm_parquet.py").exists()
        print(f"after retry: nanochat {have_tok} | lattice_atom {have_prep}")
        if not (have_tok and have_prep):
            raise SystemExit("repo checkout broken - paste this message back")

    # ------------------------------------------------------------------ 3. deps
    import importlib.util
    needs = [pkg for pkg in ("datasets", "pyarrow", "tiktoken", "rustbpe", "numpy")
             if importlib.util.find_spec(pkg) is None]
    if needs:
        print(f"installing deps: {needs} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + needs, check=True)

    # ------------------------------------------------------------------ 4. tok
    tok_dir = os.path.join(BASE_DIR, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    for f in ("tokenizer.pkl", "token_bytes.pt"):
        if not os.path.exists(os.path.join(tok_dir, f)):
            shutil.copy(os.path.join(TOKENIZER_SRC, f), os.path.join(tok_dir, f))
    if str(NANOCHAT_DIR) not in sys.path:
        sys.path.insert(0, str(NANOCHAT_DIR))
    import importlib
    tok_mod = importlib.import_module("nanochat.tokenizer")
    RustBPETokenizer = tok_mod.RustBPETokenizer
    get_token_bytes = tok_mod.get_token_bytes
    tok = RustBPETokenizer.from_directory(tok_dir)
    probe = "Lattice runs clean pretraining on a free Blackwell GPU!"
    ok = tok.decode(tok.encode(probe)) == probe
    print(f"tokenizer vocab {tok.get_vocab_size()} | roundtrip "
          f"{'OK' if ok else 'FAILED'} | bytes {tuple(get_token_bytes(device='cpu').shape)}")

    # ------------------------------------------------------------------ 5. data
    shards = os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else []
    if not shards:
        print("preparing smoke data (streams from HF, a few minutes) ...")
        log = open(os.path.join(LOG_DIR, "prep.log"), "w")
        subprocess.run(
            [sys.executable, os.path.join(LATTICE_DIR, "prepare_smollm_parquet.py"),
             "--smoke", "--out-dir", DATA_DIR],
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
        log.close()
        shards = os.listdir(DATA_DIR)
    print("shards:", sorted(shards))

    # ------------------------------------------------------------------ 6. patch
    # NOTE: sys.path must contain each package's PARENT dir:
    #   nanochat.*        -> NANOCHAT_DIR
    #   lattice_atom.*    -> REPO_DIR   (lattice_atom/ lives inside the repo root)
    for p in (str(NANOCHAT_DIR), str(REPO_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["NANOCHAT_DATA_DIR"] = DATA_DIR
    import importlib as _il
    patch_mod = _il.import_module("lattice_atom.dataset_smollm")
    _il.reload(patch_mod)
    nds = _il.import_module("nanochat.dataset")
    print(f"dataloader -> {nds.DATA_DIR} | patched: {nds.DATA_DIR == DATA_DIR}")

    # ------------------------------------------------------------------ 7. seed
    env = dict(os.environ, NANOCHAT_BASE_DIR=BASE_DIR, NANOCHAT_DATA_DIR=DATA_DIR,
               PYTHONPATH=":".join(filter(None, [LATTICE_DIR,
                                                 os.environ.get("PYTHONPATH", "")])))

    ckpt_dir = os.path.join(BASE_DIR, "base_checkpoints", "d12")
    trained = os.path.exists(os.path.join(ckpt_dir, "ckpt_00085.pt"))

    if not trained:
        print("=== PILOT PRETRAIN: depth 12, 60 steps ===")
        log = open(os.path.join(LOG_DIR, "pilot_train.log"), "w")
        p = run(["torchrun", "--standalone", "--nproc_per_node=1",
                 "-m", "scripts.base_train", "--",
                 "--run=dummy", "--depth=12", "--device-batch-size=32",
                 "--num-iterations=60", "--save-every=25", "--eval-every=50",
                 "--sample-every=-1", "--core-metric-every=-1"],
                cwd=NANOCHAT_DIR, env=env, log=log)
        t0 = time.time()
        while p.poll() is None:
            time.sleep(15)
            print(f"  pretraining ... {time.time() - t0:.0f}s", flush=True)
        log.close()
        print(f"pretrain exit {p.returncode} in {time.time() - t0:.0f}s")
        tail = subprocess.run(["tail", "-n", "20", os.path.join(LOG_DIR, "pilot_train.log")],
                              capture_output=True, text=True).stdout
        print(tail)
        if p.returncode != 0:
            print("pretrain failed - see logs/pilot_train.log. "
                  "Paste the last lines here.")
            raise SystemExit(p.returncode)

    # ------------------------------------------------------------------ 8. resume
    if not trained:
        print("=== RESUME PROOF: continue 60 -> 85 ===")
        log = open(os.path.join(LOG_DIR, "resume_train.log"), "w")
        p = run(["torchrun", "--standalone", "--nproc_per_node=1",
                 "-m", "scripts.base_train", "--",
                 "--run=dummy", "--depth=12", "--device-batch-size=32",
                 "--num-iterations=85", "--resume-from-step=60", "--save-every=25",
                 "--eval-every=-1", "--sample-every=-1", "--core-metric-every=-1"],
                cwd=NANOCHAT_DIR, env=env, log=log)
        while p.poll() is None:
            time.sleep(15)
            print("  resuming ...", flush=True)
        log.close()
        tail = subprocess.run(["tail", "-n", "10", os.path.join(LOG_DIR, "resume_train.log")],
                              capture_output=True, text=True).stdout
        print(tail)

    # ------------------------------------------------------------------ 9. verify
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
    last = torch.load(ckpts[-1], map_location="cpu", weights_only=False) if ckpts else {}
    print(f"checkpoints: {[os.path.basename(c) for c in ckpts]}")
    print(f"latest step: {last.get('step')}")
    assert last.get("step", 0) >= 85, "resume did not reach step 85"
    print("RESUME OK - the 12h-session recovery path works.")

    # ------------------------------------------------------------ 10. throughput
    log_path = os.path.join(LOG_DIR, "pilot_train.log")
    line = [_ for _ in open(log_path) if _.startswith("step ")][-1]
    print("last train line:", line.strip())
    tok_per_step = 32 * 2048
    est = tok_per_step * 60 / (time.time() - t0) if "t0" in dir() else 0
    print("(throughput math lives in the Quark 2 launch cell)")

    print()
    print("PILOT PASSED. Tell the dev machine: 'pilot passed, step 85' "
          "and ask for the Quark 2 launch command.")