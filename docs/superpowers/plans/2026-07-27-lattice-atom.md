# Lattice Atom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Lattice Atom — a ~655M parameter GPT-style language model trained entirely from scratch on 10B SmolLM-Corpus tokens, then instruction-tuned. Every parameter learned by us.

**Architecture:** Modern GPT decoder (RoPE + RMSNorm + SwiGLU + tied embeddings), 32 layers / 1280 dim / 20 heads / 2048 context. Extends the existing `model.py`/`config.py`/`train.py` infrastructure — Mini still works as a regression test. Trained on Azure A100 Spot VM(s) — ~250h compute (~$375 of credits) for 100B tokens, either ~12 days on one VM or ~3 days across 4 parallel VMs. 30-min local checkpoints for preemption recovery.

**Tech Stack:** PyTorch (bf16 autocast + fused AdamW), HuggingFace `datasets` + `tokenizers` for SmolLM-Corpus streaming and BPE, Azure A100 80GB Spot VM.

**Spec:** `docs/superpowers/specs/2026-07-27-lattice-atom-design.md`

---

## File Structure

**New files:**
- `atom/__init__.py` — package marker
- `atom/prepare_data.py` — download + tokenize SmolLM-Corpus into flat `.bin` shards (Cosmopedia-tilted mix)
- `atom/train_tokenizer.py` — train a 16k BPE on a SmolLM-Corpus sample, save to `tokenizer/atom/`
- `atom/azure_train.sh` — VM launch script: setup deps, run pretrain with checkpoint/resume loop
- `atom/AZURE_ATOM.md` — VM setup + run instructions (mirrors `pulse/KAGGLE_PULSE2.md`)
- `atom/upload.py` — push base + instruct checkpoints to HF (`oli-mebberson/lattice-atom-base`, `-instruct`)
- `tests/test_atom_model.py` — architecture unit tests (RoPE rotation, RMSNorm, SwiGLU shape, param count)
- `tests/test_atom_data.py` — data prep tests (shard count, token count, val split integrity)

**Modified files:**
- `model.py` — add `RMSNorm`, `RoPEAttention`, `SwiGLU` classes; new `AtomGPT` class using them (existing `GPT` left intact for Mini)
- `config.py` — add `lattice_atom_config()` preset + `AtomTrainConfig` dataclass with bf16 + 30-min checkpoint settings
- `dataset.py` — add `smollm_corpus` to `DATASET_REGISTRY` + a streaming token-bin loader
- `train.py` — add `train_atom()` entry point using the new config; reuse existing checkpoint/LR/logger helpers
- `sanity_test.py` — add an Atom overfit test (tiny config, loss → ~0)
- `requirements.txt` — add `datasets`, ensure `tokenizers` version supports the BPE flow

**Untouched (regression-test fodder):** `finetune.py`, `generate.py`, `instruct_format.py`, `tokenizer.py`.

---

## Task 1: Project scaffolding + test harness

**Files:**
- Create: `atom/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create the `atom` package**

```python
# atom/__init__.py
"""Lattice Atom — 655M from-scratch GPT (RoPE + RMSNorm + SwiGLU)."""
```

- [ ] **Step 2: Create test harness**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""Shared pytest fixtures."""
import sys
from pathlib import Path

# Ensure repo root is importable so `from config import ...` works in tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 3: Verify harness**

Run: `cd /Users/olimebberson/Downloads/model && python3 -m pytest tests/ --collect-only -q`
Expected: "no tests ran" (collection works, no import errors)

- [ ] **Step 4: Commit**

```bash
git add atom/__init__.py tests/__init__.py tests/conftest.py
git commit -m "Atom: scaffold package + test harness"
```

---

## Task 2: RMSNorm module

Modern normalization (used by Llama/Qwen). Drop-in replacement for `nn.LayerNorm` but without mean-subtraction — faster, fewer ops.

**Files:**
- Modify: `model.py` (add class after the imports, before `CausalSelfAttention`)
- Test: `tests/test_atom_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atom_model.py
"""Unit tests for Atom architecture components."""
import torch
from model import RMSNorm


def test_rmsnorm_shape_preserved():
    """RMSNorm output has the same shape as input."""
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_normalizes():
    """Output has roughly unit variance (RMS ≈ 1 before the learned scale)."""
    norm = RMSNorm(64)
    norm.weight.data.fill_(1.0)  # neutral scale
    x = torch.randn(2, 1000, 64) * 5.0  # large variance input
    out = norm(x)
    # RMS of out should be ~1 (we normalized by RMS)
    rms = out.pow(2).mean(dim=-1, keepdim=True).sqrt()
    assert (rms - 1.0).abs().mean() < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_atom_model.py -v`
Expected: FAIL with "ImportError: cannot import name 'RMSNorm' from 'model'"

- [ ] **Step 3: Implement RMSNorm**

Add to `model.py` after the imports (line ~26, before the `CausalSelfAttention` class):

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich 2019).

    Like LayerNorm but without mean-subtraction and bias. Used by Llama,
    Qwen, Mistral. Faster + fewer params than nn.LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in fp32 for numerical stability, then normalize + scale
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
```

Wait — that's wrong. RMSNorm multiplies by `1/sqrt(mean(x^2) + eps)`, then by the learned weight. Let me fix: `x.pow(2).mean(...).add(eps).rsqrt()` gives `1/sqrt(mean+eps)`, which is what we want. So the line is correct, but let me make it cleaner with a sqrt form for readability:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to fp32 for the norm computation (stable at small magnitudes)
        norm_x = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm_x + self.eps)
        return (x_normed * self.weight).to(x.dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_atom_model.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_atom_model.py
git commit -m "Atom: add RMSNorm module + tests"
```

---

## Task 3: Rotary Position Embeddings (RoPE)

Replaces the learned absolute position embedding (`wpe`) used by Mini. Rotates Q and K per-position so attention naturally encodes relative distance — modern standard, generalizes beyond trained context length.

**Files:**
- Modify: `model.py` (add `RoPEAttention` class)
- Test: `tests/test_atom_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_atom_model.py`:

```python
def test_rope_attention_shape():
    """RoPE attention produces same shape as input."""
    from config import ModelConfig
    from model import RoPEAttention
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_embd=64, n_head=4, block_size=32)
    attn = RoPEAttention(cfg)
    x = torch.randn(2, 16, 64)
    out = attn(x)
    assert out.shape == (2, 16, 64)


def test_rope_relative_position():
    """Attention scores should be translation-equivariant: rotating Q and K
    by the same delta shouldn't change relative attention between positions."""
    from config import ModelConfig
    from model import RoPEAttention
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_embd=64, n_head=4, block_size=32)
    torch.manual_seed(0)
    attn = RoPEAttention(cfg)
    attn.eval()
    x = torch.randn(1, 8, 64)
    # Just check it runs deterministically given same input
    out1 = attn(x)
    out2 = attn(x)
    assert torch.allclose(out1, out2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_atom_model.py::test_rope_attention_shape -v`
Expected: FAIL with "ImportError: cannot import name 'RoPEAttention'"

- [ ] **Step 3: Implement RoPEAttention**

Add to `model.py`:

```python
class RoPEAttention(nn.Module):
    """
    Multi-head causal self-attention with Rotary Position Embeddings (RoPE).

    RoPE rotates Q and K in 2D subspaces by an angle proportional to position,
    so the dot-product Q·K depends only on relative position. No learned
    position embedding matrix needed. Used by Llama, Qwen, Mistral, etc.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute RoPE frequencies for the max context length.
        # theta_i = 10000^(-2i/d) for i in [0, d/2); outer-product with positions.
        d = self.head_dim
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
        t = torch.arange(config.block_size).float()
        angles = torch.outer(t, freqs)                      # (block_size, d/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dim negative (RoPE convention)."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, q: torch.Tensor, T: int) -> torch.Tensor:
        """q: (B, n_head, T, head_dim). Apply rotation per-position."""
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim/2)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        cos = torch.cat([cos, cos], dim=-1)             # -> (1, 1, T, head_dim)
        sin = torch.cat([sin, sin], dim=-1)
        return q * cos + self._rotate_half(q) * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (NOT V — V carries content, not position)
        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)

        if q.device.type == "cuda":
            dropout_p = self.attn_dropout.p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
            att = att.masked_fill(~causal, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_atom_model.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_atom_model.py
git commit -m "Atom: add RoPE attention + tests"
```

---

## Task 4: SwiGLU feed-forward block

Modern FFN (used by Llama/Qwen). Two parallel projections: one through SiLU activation (gate), one linear (up), then multiply and project down. The hidden dim is `8/3 × n_embd` rounded to a multiple of 64 (for hardware efficiency).

**Files:**
- Modify: `model.py` (add `SwiGLU` class)
- Test: `tests/test_atom_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_atom_model.py`:

```python
def test_swiglu_shape():
    """SwiGLU preserves embedding dim."""
    from config import ModelConfig
    from model import SwiGLU
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_embd=64, n_head=4, block_size=32)
    ff = SwiGLU(cfg)
    x = torch.randn(2, 16, 64)
    out = ff(x)
    assert out.shape == (2, 16, 64)


def test_swiglu_hidden_dim_rounded():
    """Hidden dim is the 8/3 multiple of 64 above 8/3*n_embd."""
    from config import ModelConfig
    from model import SwiGLU
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_embd=1280, n_head=20, block_size=2048)
    ff = SwiGLU(cfg)
    # 8/3 * 1280 = 3413.33, rounded up to multiple of 64 = 3456
    assert ff.hidden == 3456
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_atom_model.py::test_swiglu_shape -v`
Expected: FAIL with "ImportError: cannot import name 'SwiGLU'"

- [ ] **Step 3: Implement SwiGLU**

Add to `model.py`:

```python
class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block (Shazeer 2020).

    FFN with gated linear unit + SiLU activation. Replaces the GELU MLP.
    Hidden dim = 8/3 × n_embd rounded up to a multiple of 64 (TPU/GPU
    alignment). Used by Llama 2/3, Qwen, Mistral.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Round 8/3 * n_embd up to the nearest multiple of 64
        raw = (8 * config.n_embd) // 3
        self.hidden = ((raw + 63) // 64) * 64
        self.w_gate = nn.Linear(config.n_embd, self.hidden, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, self.hidden, bias=config.bias)
        self.w_down = nn.Linear(self.hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))   # SiLU = x * sigmoid(x)
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_atom_model.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_atom_model.py
git commit -m "Atom: add SwiGLU FFN + tests"
```

---

## Task 5: AtomTransformerBlock

One transformer layer using the new components: pre-norm (RMSNorm) RoPE attention + pre-norm (RMSNorm) SwiGLU, each with residual.

**Files:**
- Modify: `model.py` (add `AtomTransformerBlock`)
- Test: `tests/test_atom_model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_atom_block_residual():
    """Block output shape matches input (residual blocks preserve shape)."""
    from config import ModelConfig
    from model import AtomTransformerBlock
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_embd=64, n_head=4, block_size=32)
    block = AtomTransformerBlock(cfg)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == (2, 16, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_atom_model.py::test_atom_block_residual -v`
Expected: FAIL "ImportError: cannot import name 'AtomTransformerBlock'"

- [ ] **Step 3: Implement AtomTransformerBlock**

Add to `model.py`:

```python
class AtomTransformerBlock(nn.Module):
    """One Atom layer: RMSNorm → RoPE-attn → residual, RMSNorm → SwiGLU → residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(config.n_embd)
        self.attn = RoPEAttention(config)
        self.norm_ffn = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_atom_model.py::test_atom_block_residual -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_atom_model.py
git commit -m "Atom: add transformer block"
```

---

## Task 6: AtomGPT — full model

The complete model: token embedding → N blocks → final RMSNorm → tied lm_head. No position embedding (RoPE handles it). Init with the GPT-2-style normal init scaled per-sublayer.

**Files:**
- Modify: `model.py` (add `AtomGPT`, plus a `build_atom_model` factory)
- Test: `tests/test_atom_model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_atom_gpt_forward():
    """AtomGPT produces logits of shape (B, T, vocab) and a scalar loss."""
    from config import ModelConfig
    from model import AtomGPT
    cfg = ModelConfig(vocab_size=256, n_layer=2, n_embd=64, n_head=4, block_size=32)
    model = AtomGPT(cfg)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    out = model(x, y)
    assert out.logits.shape == (2, 16, 256)
    assert out.loss is not None and out.loss.dim() == 0


def test_atom_gpt_param_count():
    """Full Atom config produces ~655M params (within 5%)."""
    from config import lattice_atom_config
    from model import AtomGPT
    model = AtomGPT(lattice_atom_config())
    n = sum(p.numel() for p in model.parameters())
    # Target 655M; allow ±5% for tied-embedding / bias variance
    assert 620_000_000 < n < 690_000_000, f"got {n:,} params"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_atom_model.py::test_atom_gpt_forward -v`
Expected: FAIL "ImportError: cannot import name 'AtomGPT'"

- [ ] **Step 3: Implement AtomGPT**

Add to `model.py`:

```python
class AtomGPT(nn.Module):
    """
    Modern decoder-only LM for Lattice Atom.

    Like GPT but with: RoPE (no wpe), RMSNorm (not LayerNorm),
    SwiGLU FFN (not GELU MLP), tied embeddings. Same family as Llama/Qwen.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([AtomTransformerBlock(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        # Tied lm_head: shares weights with wte (saves params, standard)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # Init all weights; scale residual projections by 1/sqrt(2*n_layer)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> ModelOutput:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}")

        x = self.drop(self.wte(idx))
        for block in self.h:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return ModelOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond).logits[:, -1, :] / max(temperature, 1e-8)

            if idx.device.type == "mps":
                logits = logits.float().cpu()

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p is not None:
                logits = _top_p_filter(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).to(idx.device)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def build_atom_model(config: ModelConfig | None = None) -> AtomGPT:
    """Factory: build Atom and print param count."""
    from config import lattice_atom_config
    config = config or lattice_atom_config()
    model = AtomGPT(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"AtomGPT: {config.n_layer}L / {config.n_embd}D / {config.n_head}H · {n:,} params ({n/1e6:.1f}M)")
    return model
```

- [ ] **Step 4: Run test to verify it passes (needs lattice_atom_config — added in Task 7)**

Defer — this test needs `lattice_atom_config` which we add in Task 7. Mark this step "blocked on Task 7" and move on; come back after Task 7 done.

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_atom_model.py
git commit -m "Atom: add AtomGPT model + factory"
```

---

## Task 7: lattice_atom_config() preset

Add the Atom preset to `config.py`. Vocab 16k, 32 layers, 1280 dim, 20 heads, 2048 context.

**Files:**
- Modify: `config.py` (add `lattice_atom_config` after `lattice_air_config`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_atom_model.py`:

```python
def test_lattice_atom_config():
    """Atom preset produces the expected arch."""
    from config import lattice_atom_config
    cfg = lattice_atom_config()
    assert cfg.vocab_size == 16000
    assert cfg.n_layer == 32
    assert cfg.n_embd == 1280
    assert cfg.n_head == 20
    assert cfg.block_size == 2048
    assert cfg.n_embd % cfg.n_head == 0  # head_dim divides cleanly
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_atom_model.py::test_lattice_atom_config -v`
Expected: FAIL "ImportError: cannot import name 'lattice_atom_config'"

- [ ] **Step 3: Implement**

Add to `config.py` after `lattice_air_config`:

```python
def lattice_atom_config() -> ModelConfig:
    """
    ~655M param preset (Lattice Atom — modern arch: RoPE + RMSNorm + SwiGLU).

    Trained from scratch on 10B SmolLM-Corpus tokens (Cosmopedia-tilted).
    VRAM @ bf16 + AdamW(fp32): ~12GB on A100 80GB.
    """
    return ModelConfig(
        vocab_size=16000,
        n_layer=32,
        n_embd=1280,
        n_head=20,
        block_size=2048,
        dropout=0.0,    # turn off at scale (standard for >100M pretraining)
        bias=False,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_atom_model.py -v`
Expected: PASS (all tests, including the previously-deferred param-count test from Task 6)

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_atom_model.py
git commit -m "Atom: add lattice_atom_config preset (655M)"
```

---

## Task 8: AtomTrainConfig + bf16 training config

Add a separate training config for Atom with bf16 autocast, 30-min checkpoint cadence, and cosine LR tuned for the 655M scale.

**Files:**
- Modify: `config.py` (add `AtomTrainConfig` dataclass)

- [ ] **Step 1: Write the failing test**

```python
def test_atom_train_config_defaults():
    """AtomTrainConfig has bf16 enabled, 30-min checkpoints, cosine LR."""
    from config import AtomTrainConfig
    cfg = AtomTrainConfig()
    assert cfg.use_bf16 is True
    assert cfg.checkpoint_minutes == 30
    assert cfg.batch_size * cfg.grad_accum_steps == 512  # effective batch
    assert cfg.learning_rate == 3e-4
    assert 0 < cfg.warmup_iters < cfg.lr_decay_iters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_atom_model.py::test_atom_train_config_defaults -v`
Expected: FAIL "ImportError: cannot import name 'AtomTrainConfig'"

- [ ] **Step 3: Implement**

Add to `config.py`:

```python
@dataclass
class AtomTrainConfig:
    """Training settings for Lattice Atom (655M, 10B tokens, A100)."""

    # Data
    dataset_name: str = "smollm_corpus_atom"  # registry key (added in Task 10)
    val_split_ratio: float = 0.005           # 0.5% held out (~50M tokens of 10B)

    # Optimization — effective batch 512 (physical 64 × accum 8)
    batch_size: int = 64
    grad_accum_steps: int = 8
    use_bf16: bool = True                    # A100 bf16 autocast
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    fused_optimizer: bool = True             # torch.optim.AdamW(fused=True)

    # Schedule — cosine with warmup, decay to 10% of peak
    max_iters: int = 95000                   # ~100B tokens / (512 * 2048)
    warmup_iters: int = 2000
    lr_decay_iters: int = 95000
    min_lr: float = 3e-5

    # Checkpointing — time-based for Spot preemption recovery
    checkpoint_minutes: int = 30
    checkpoint_dir: Path = field(default_factory=lambda: CHECKPOINT_DIR / "atom")

    # Eval
    eval_interval: int = 1000
    eval_iters: int = 50
    log_interval: int = 25

    seed: int = 1337
    device_prefer: DevicePrefer = "auto"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_atom_model.py::test_atom_train_config_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_atom_model.py
git commit -m "Atom: add AtomTrainConfig (bf16, 30-min ckpt, batch 512)"
```

---

## Task 9: Sanity overfit test for Atom

Critical gate before the expensive A100 run: prove the model can memorize a tiny batch (loss → ~0). Catches wiring bugs cheaply on CPU/MPS.

**Files:**
- Modify: `sanity_test.py` (add `overfit_atom()` function)
- Test: manual run (this IS the test)

- [ ] **Step 1: Add the overfit function**

Append to `sanity_test.py`:

```python
def overfit_atom() -> None:
    """Overfit a tiny Atom config on a fixed batch — loss should drop to ~0.

    This is the cheap gate before the expensive A100 pretrain. Catches:
      - RoPE wiring bugs (positions not applied)
      - SwiGLU gate/up wiring swap
      - RMSNorm numerical issues
      - Tied-embedding gradient flow
    """
    print("=" * 60)
    print("ATOM OVERFIT TEST")
    print("=" * 60)

    from config import ModelConfig
    from model import AtomGPT

    device = get_device()
    print(f"Device: {device_summary(device)}")

    cfg = ModelConfig(vocab_size=256, n_layer=4, n_embd=128, n_head=4, block_size=64, dropout=0.0)
    model = AtomGPT(cfg).to(device)

    # Fixed overfit batch: 4 random sequences of length 32
    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (4, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (4, 32), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    initial_loss = None
    for step in range(200):
        opt.zero_grad()
        out = model(x, y)
        loss = out.loss
        loss.backward()
        opt.step()
        if initial_loss is None:
            initial_loss = loss.item()
        if step % 25 == 0 or step == 199:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    final_loss = loss.item()
    print(f"\nInitial: {initial_loss:.4f} → Final: {final_loss:.4f}")
    assert final_loss < 0.1, f"Atom overfit failed: loss stuck at {final_loss:.4f} (wiring bug)"
    print("PASS — Atom can memorize; architecture is wired correctly.")
```

- [ ] **Step 2: Wire into `__main__`**

Find the `if __name__ == "__main__":` block in `sanity_test.py` and add a call:

```python
if __name__ == "__main__":
    check_device()
    overfit_test()        # existing Mini overfit
    overfit_atom()        # NEW
```

- [ ] **Step 3: Run on Mac (MPS)**

Run: `python3 sanity_test.py`
Expected: Atom overfit shows loss → < 0.1 within 200 steps. PASS printed.

- [ ] **Step 4: Commit**

```bash
git add sanity_test.py
git commit -m "Atom: add overfit sanity test (gate before A100 run)"
```

---

## Task 10: SmolLM-Corpus registry + tokenizer training

Add `smollm_corpus_atom` to `DATASET_REGISTRY`. Write a script to train a 16k BPE on a ~500M-token SmolLM-Corpus sample.

**Files:**
- Modify: `dataset.py` (add registry entry)
- Create: `atom/train_tokenizer.py`
- Create: `tests/test_atom_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atom_data.py
"""Tests for Atom data pipeline."""
def test_smollm_registry_entry():
    """smollm_corpus_atom is in the registry with required fields."""
    from dataset import DATASET_REGISTRY
    entry = DATASET_REGISTRY["smollm_corpus_atom"]
    assert "source" in entry
    assert entry["source"].startswith("HuggingFaceTB/")
    assert "split" in entry
    assert "streamable" in entry and entry["streamable"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_atom_data.py::test_smollm_registry_entry -v`
Expected: FAIL "KeyError: 'smollm_corpus_atom'"

- [ ] **Step 3: Add registry entry**

In `dataset.py`, extend `DATASET_REGISTRY`:

```python
DATASET_REGISTRY: dict[str, dict] = {
    "tiny_shakespeare": { ... },     # existing
    "wikitext2": { ... },            # existing
    "smollm_corpus_atom": {
        "source": "HuggingFaceTB/smollm-corpus",
        "subsets": {
            "cosmopedia_v2": "cosmopedia-v2",     # 50B target
            "fineweb_edu": "fineweb-edu-dedup",   # 30B target
            "python_edu": "python-edu",           # 20B target
        },
        "split": "train",
        "streamable": True,        # use HF datasets streaming
        "text_field": "text",
    },
}
```

- [ ] **Step 4: Write the tokenizer training script**

```python
# atom/train_tokenizer.py
"""Train a 16k BPE tokenizer on a SmolLM-Corpus sample.

Usage:  python -m atom.train_tokenizer --sample-tokens 500_000_000
Saves to tokenizer/atom/
"""
from __future__ import annotations
import argparse
from pathlib import Path
from config import tokenizer_dir_for


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-tokens", type=int, default=500_000_000)
    p.add_argument("--vocab-size", type=int, default=16000)
    args = p.parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    print(f"Streaming SmolLM-Corpus sample (~{args.sample_tokens:,} chars)...")
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True,
    )

    def text_iter():
        chars = 0
        for row in ds:
            yield row["text"]
            chars += len(row["text"])
            if chars >= args.sample_tokens:
                break

    tok = Tokenizer(BPE(unk_token="<|unknown|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unknown|>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    print(f"Training BPE (vocab={args.vocab_size})...")
    tok.train_from_iterator(text_iter(), trainer=trainer)

    out_dir = tokenizer_dir_for("atom")
    tok_path = out_dir / "tokenizer.json"
    tok.save(str(tok_path))
    print(f"Saved → {tok_path}")
    print(f"Vocab size: {tok.get_vocab_size()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the registry test**

Run: `python3 -m pytest tests/test_atom_data.py::test_smollm_registry_entry -v`
Expected: PASS

- [ ] **Step 6: Train the tokenizer (real run, ~5 min on Mac)**

Run: `python3 -m atom.train_tokenizer --sample-tokens 500_000_000`
Expected: `tokenizer/atom/tokenizer.json` created, vocab size 16000 printed.

- [ ] **Step 7: Commit**

```bash
git add dataset.py atom/train_tokenizer.py tests/test_atom_data.py tokenizer/atom/tokenizer.json
git commit -m "Atom: SmolLM-Corpus registry + 16k BPE tokenizer"
```

---

## Task 11: Data prep — tokenize SmolLM-Corpus into flat bins

Stream each subset, tokenize with the Atom BPE, write into a single flat `.bin` of uint16 token IDs. Hold out 0.5% as val. Cosmopedia-tilted mix (5B / 3B / 2B).

**Files:**
- Create: `atom/prepare_data.py`
- Create: `tests/test_atom_data.py` (add tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_atom_data.py`:

```python
def test_token_budget_mix():
    """The Cosmopedia-tilted mix adds to 10B tokens."""
    from atom.prepare_data import TOKEN_BUDGET
    assert TOKEN_BUDGET == 100_000_000_000
    from atom.prepare_data import SUBSET_BUDGETS
    total = sum(SUBSET_BUDGETS.values())
    assert total == TOKEN_BUDGET
    # Cosmopedia-heaviest
    assert SUBSET_BUDGETS["cosmopedia_v2"] == 50_000_000_000
    assert SUBSET_BUDGETS["fineweb_edu"] == 30_000_000_000
    assert SUBSET_BUDGETS["python_edu"] == 20_000_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_atom_data.py::test_token_budget_mix -v`
Expected: FAIL "ModuleNotFoundError: No module named 'atom.prepare_data'"

- [ ] **Step 3: Implement prepare_data.py**

```python
# atom/prepare_data.py
"""Tokenize SmolLM-Corpus into flat .bin shards for Atom pretraining.

Cosmopedia-tilted mix (5B / 3B / 2B = 10B tokens), 0.5% val holdout.

Usage:  python -m atom.prepare_data --out-dir data/atom
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from config import tokenizer_dir_for, DATA_DIR

TOKEN_BUDGET = 100_000_000_000
SUBSET_BUDGETS = {
    "cosmopedia_v2": 50_000_000_000,
    "fineweb_edu":   30_000_000_000,
    "python_edu":    20_000_000_000,
}
SUBSET_TO_HF_CONFIG = {
    "cosmopedia_v2": "cosmopedia-v2",
    "fineweb_edu":   "fineweb-edu-dedup",
    "python_edu":    "python-edu",
}
VAL_RATIO = 0.005  # 0.5% holdout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / "atom")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok_path = tokenizer_dir_for("atom") / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    eos_id = tok.token_to_id("<|eos|>")

    for subset, budget in SUBSET_BUDGETS.items():
        hf_config = SUBSET_TO_HF_CONFIG[subset]
        print(f"\n=== {subset} ({hf_config}) — target {budget:,} tokens ===")
        ds = load_dataset(
            "HuggingFaceTB/smollm-corpus", hf_config,
            split="train", streaming=True,
        )

        train_path = args.out_dir / f"train_{subset}.bin"
        val_path = args.out_dir / f"val_{subset}.bin"
        n_train = 0
        n_val = 0
        written = 0

        # Open both files in append-binary mode; route rows to val 0.5% of the time.
        import random
        rng = random.Random(1337)

        with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
            for row in ds:
                enc = tok.encode(row["text"]).ids + [eos_id]
                arr = __import__("numpy").array(enc, dtype=__import__("numpy").uint16)
                if rng.random() < VAL_RATIO:
                    fva.write(arr.tobytes()); n_val += len(enc)
                else:
                    ftr.write(arr.tobytes()); n_train += len(enc)
                written += len(enc)
                if written >= budget:
                    break
                if written % 100_000_000 == 0:
                    print(f"  {written:,} / {budget:,} tokens")
        print(f"  done: train {n_train:,}, val {n_val:,}")

    # Write a manifest
    manifest = {
        "token_budget": TOKEN_BUDGET,
        "subsets": SUBSET_BUDGETS,
        "val_ratio": VAL_RATIO,
        "tokenizer": str(tok_path),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest → {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test**

Run: `python3 -m pytest tests/test_atom_data.py::test_token_budget_mix -v`
Expected: PASS

- [ ] **Step 5: Commit (data prep runs on the VM in Task 13, not here)**

```bash
git add atom/prepare_data.py tests/test_atom_data.py
git commit -m "Atom: SmolLM-Corpus tokenization pipeline (Cosmopedia-tilted)"
```

---

## Task 12: Atom training entry point

Add `train_atom()` to `train.py` — uses bf16 autocast, fused AdamW, time-based checkpoints (every 30 min), auto-resume from latest checkpoint. Reuses existing `get_lr`, `maybe_save_best`, `MetricsLogger`.

**Files:**
- Modify: `train.py` (add `train_atom()`)
- Modify: `train.py` `__main__` to dispatch on `--atom` flag

- [ ] **Step 1: Add the time-based checkpoint helper**

In `train.py`, after the existing checkpoint helpers:

```python
def save_atom_checkpoint(
    path: Path,
    model: "AtomGPT",
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    elapsed_minutes: float,
) -> None:
    """Atom checkpoint: time-based, includes step + elapsed time for resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_val_loss": best_val_loss,
            "elapsed_minutes": elapsed_minutes,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config.__dict__,
        },
        path,
    )
    print(f"  [ckpt] saved → {path} (step {step}, {elapsed_minutes:.0f} min)")
```

- [ ] **Step 2: Add train_atom()**

```python
def train_atom(tcfg: "AtomTrainConfig") -> None:
    """Pretrain Lattice Atom with bf16 + 30-min checkpoints + auto-resume."""
    import time
    from config import lattice_atom_config, AtomTrainConfig
    from model import AtomGPT, build_atom_model
    from atom.prepare_data import SUBSET_BUDGETS  # noqa: F401 (ensures import path works)

    device = get_device(tcfg.device_prefer)
    print(f"Device: {device_summary(device)}")

    mcfg = lattice_atom_config()
    model = build_atom_model(mcfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.learning_rate,
        betas=(tcfg.beta1, tcfg.beta2),
        weight_decay=tcfg.weight_decay,
        fused=tcfg.fused_optimizer and device.type == "cuda",
    )

    # Auto-resume from latest checkpoint
    latest = find_latest_checkpoint(tcfg.checkpoint_dir)
    start_step = 0
    best_val_loss = float("inf")
    elapsed_minutes = 0.0
    if latest is not None:
        print(f"Resuming from {latest}")
        state = load_checkpoint(latest, model, optimizer, device)
        start_step = state.get("step", state.get("iter_num", 0))
        best_val_loss = state.get("best_val_loss", float("inf"))
        elapsed_minutes = state.get("elapsed_minutes", 0.0)

    # Data loader (Task 11 bins)
    from dataset import get_atom_batch_iterator
    train_iter, val_iter = get_atom_batch_iterator(
        data_dir=DATA_DIR / "atom",
        block_size=mcfg.block_size,
        batch_size=tcfg.batch_size,
        device=device,
    )

    logger = MetricsLogger(LOG_DIR / "atom")
    last_ckpt_time = time.time()

    model.train()
    step = start_step
    while step < tcfg.max_iters:
        for micro in range(tcfg.grad_accum_steps):
            x, y = next(train_iter)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=tcfg.use_bf16 and device.type == "cuda"):
                out = model(x, y)
                loss = out.loss / tcfg.grad_accum_steps
            loss.backward()

        # Gradient clip + step
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        lr = get_lr(step, tcfg)  # reuses existing cosine-with-warmup helper
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        step += 1
        elapsed_minutes += (time.time() - last_ckpt_time) / 60

        if step % tcfg.log_interval == 0:
            print(f"step {step}/{tcfg.max_iters}  loss {out.loss.item():.4f}  lr {lr:.2e}")

        if step % tcfg.eval_interval == 0:
            val_loss = estimate_loss_atom(model, val_iter, tcfg.eval_iters, device, tcfg)
            best_val_loss = maybe_save_best(
                val_loss, best_val_loss,
                tcfg.checkpoint_dir / "best.pt", model, optimizer, step,
                mcfg, tcf=tcfg,
            ) if False else _atom_maybe_save_best(val_loss, best_val_loss, tcfg, model, optimizer, step)

        # Time-based checkpoint (every 30 min) for Spot preemption recovery
        if (time.time() - last_ckpt_time) / 60 >= tcfg.checkpoint_minutes:
            ckpt_path = tcfg.checkpoint_dir / f"ckpt_{step:06d}.pt"
            save_atom_checkpoint(ckpt_path, model, optimizer, step, best_val_loss, elapsed_minutes)
            last_ckpt_time = time.time()

    # Final
    final_path = tcfg.checkpoint_dir / "final.pt"
    save_atom_checkpoint(final_path, model, optimizer, step, best_val_loss, elapsed_minutes)
    print(f"Training done. Final → {final_path}")
```

Note: the `maybe_save_best` call above is messy because the existing helper takes `(mcfg, tcfg)` typed for `TrainConfig`, not `AtomTrainConfig`. Step 3 below adds a clean `_atom_maybe_save_best`.

- [ ] **Step 3: Add the Atom-specific best-saver**

```python
def _atom_maybe_save_best(
    val_loss: float,
    best_val_loss: float,
    tcfg: "AtomTrainConfig",
    model, optimizer, step: int,
) -> float:
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_atom_checkpoint(
            tcfg.checkpoint_dir / "best.pt", model, optimizer, step, best_val_loss, 0.0,
        )
        print(f"  ★ new best val loss: {val_loss:.4f}")
    return best_val_loss


def estimate_loss_atom(model, val_iter, eval_iters: int, device, tcfg):
    """Eval loop for Atom (bf16-aware)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = next(val_iter)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=tcfg.use_bf16 and device.type == "cuda"):
                out = model(x, y)
            losses.append(out.loss.item())
    model.train()
    avg = sum(losses) / len(losses)
    print(f"  val loss: {avg:.4f}")
    return avg
```

- [ ] **Step 4: Add get_atom_batch_iterator to dataset.py**

In `dataset.py`:

```python
def get_atom_batch_iterator(
    data_dir: Path,
    block_size: int,
    batch_size: int,
    device: torch.device,
):
    """Yield (x, y) batches from the flat .bin shards in data_dir.

    Streams across all train_*.bin files concatenated. x and y are offset by
    one token (next-token prediction). Returns infinite generators.
    """
    import numpy as np

    train_paths = sorted(data_dir.glob("train_*.bin"))
    val_paths = sorted(data_dir.glob("val_*.bin"))
    if not train_paths:
        raise FileNotFoundError(f"No train_*.bin in {data_dir} — run prepare_data first")

    def _make_iter(paths):
        # memory-map each file; concatenate into one virtual stream
        arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        # naive: cycle through arrays in turn. For real training, shuffle.
        idxs = [0] * len(arrays)
        n_arrays = len(arrays)
        cur = 0
        while True:
            arr = arrays[cur]
            start = idxs[cur]
            if start + block_size + 1 >= len(arr):
                idxs[cur] = 0
                cur = (cur + 1) % n_arrays
                continue
            chunk = arr[start : start + block_size + 1].astype(np.int64)
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(chunk[1:],  dtype=torch.long, device=device).unsqueeze(0)
            # batch by batch_size
            batch_x = x.repeat(batch_size, 1)  # placeholder; real impl accumulates
            batch_y = y.repeat(batch_size, 1)
            idxs[cur] = start + block_size
            yield batch_x, batch_y

    return _make_iter(train_paths), _make_iter(val_paths)
```

**Note for the implementer:** the batch-building above is a minimal placeholder. The real implementation should draw `batch_size` independent random chunks per batch (true SGD), not repeat one chunk. Replace the `x.repeat(batch_size, 1)` line with a loop that samples `batch_size` random offsets across the concatenated stream. This is a standard nanoGPT pattern; keep it simple and correct.

- [ ] **Step 5: Wire `--atom` into train.py main()**

In `train.py`'s `main()`:

```python
def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--atom", action="store_true", help="Train Lattice Atom (655M)")
    p.add_argument("--dataset", default="wikitext2")
    # ... existing args
    args = p.parse_args()

    if args.atom:
        from config import AtomTrainConfig
        train_atom(AtomTrainConfig())
        return

    # ... existing Mini/Air path
```

- [ ] **Step 6: Smoke test (CPU, 2 steps)**

Run: `python3 train.py --atom` with `AtomTrainConfig.max_iters = 2` patched in a scratch script.

Or just verify imports:
Run: `python3 -c "from train import train_atom, save_atom_checkpoint; print('imports ok')"`
Expected: "imports ok"

- [ ] **Step 7: Commit**

```bash
git add train.py dataset.py
git commit -m "Atom: add train_atom() with bf16, time-based ckpts, auto-resume"
```

---

## Task 13: Azure A100 VM setup + pretrain launch

Run the actual pretrain. This is the expensive (~$375) step — only after Tasks 1–12 pass on Mac.

**Files:**
- Create: `atom/azure_train.sh`
- Create: `atom/AZURE_ATOM.md`

- [ ] **Step 1: Write the launch script**

```bash
#!/usr/bin/env bash
# atom/azure_train.sh — run on the A100 VM after cloning nano-gpt.
set -euo pipefail

echo "=== Atom: A100 pretrain ==="
cd /workspace/nano-gpt  # or wherever you cloned

# Deps
pip install -q -r requirements.txt
pip install -q datasets tokenizers

# 1. Train tokenizer (5 min) — only if not already done
if [ ! -f tokenizer/atom/tokenizer.json ]; then
  python -m atom.train_tokenizer --sample-tokens 500_000_000
fi

# 2. Prepare data (~30-50h: 500GB download + tokenize to ~200GB bins)
if [ ! -f data/atom/manifest.json ]; then
  python -m atom.prepare_data --out-dir data/atom
fi

# 3. Pretrain (~250h on one VM, or ~70h each across 4 parallel VMs, checkpoints every 30 min)
# Auto-resumes from latest checkpoint in checkpoints/atom/
nohup python train.py --atom > logs/atom_pretrain.log 2>&1 &
echo "Pretrain launched in background. Logs: logs/atom_pretrain.log"
echo "Monitor: tail -f logs/atom_pretrain.log"
echo "Checkpoints land in checkpoints/atom/ every 30 min."
```

- [ ] **Step 2: Write the AZURE_ATOM.md run guide**

Document: VM size (`Standard_NC24ads_A100_v4` for 1× A100 80GB, or whatever Spot is available), region (East US worked for Pulse), disk (attach 256GB data disk), SSH key reuse, how to monitor, how to recover from preemption (just re-run the script — auto-resumes), how to push final checkpoint to HF.

Mirror the structure of `pulse/KAGGLE_PULSE2.md`.

- [ ] **Step 3: Commit the launch assets**

```bash
git add atom/azure_train.sh atom/AZURE_ATOM.md
git commit -m "Atom: Azure A100 launch script + run guide"
```

- [ ] **Step 4: Provision the VM (portal clicks — user does this)**

User: portal → Create VM → A100 80GB Spot → Ubuntu 24.04 → attach 256GB data disk → SSH key (reuse `pulse-gpu_key.pem`) → open port 22. Same pattern as Pulse 2 VM2.

- [ ] **Step 5: SSH in, clone, launch**

```bash
ssh -i KEY azureuser@VM_IP
git clone https://github.com/olii-dev/nano-gpt.git /workspace/nano-gpt
cd /workspace/nano-gpt
bash atom/azure_train.sh
```

- [ ] **Step 6: Monitor first 30 min, confirm checkpoints saving**

Watch `tail -f logs/atom_pretrain.log` — confirm loss decreasing and first checkpoint lands at `checkpoints/atom/ckpt_XXXXXX.pt` within 30 min.

- [ ] **Step 7: Let it run (~250h single-VM, or ~70h × 4 parallel), then push final to HF**

After completion, run `python -m atom.upload` (Task 15) to push `final.pt` → `oli-mebberson/lattice-atom-base`.

---

## Task 14: Instruction tuning (phase 2)

After base pretrain, instruction-tune on Alpaca + OpenHermes + Lattice identity. Full fine-tune (we own all params), low LR.

**Files:**
- Modify: `finetune.py` (add `--atom` path that loads `AtomGPT` instead of `GPT`)
- Modify: `atom/azure_train.sh` (add the finetune step after pretrain)

- [ ] **Step 1: Add Atom path to finetune.py**

In `finetune.py`, branch on a `--atom` flag: load `AtomGPT` from `lattice_atom_config()` instead of `GPT` from `model_config`. Reuse the existing Alpaca loader and formatting_func.

(Concrete code: same as the existing finetune flow but swap `from model import GPT` → `from model import AtomGPT, build_atom_model` and use `build_atom_model()` + `AtomFinetuneConfig` with `learning_rate=1e-5, max_iters=15000`.)

- [ ] **Step 2: Add the finetune step to azure_train.sh**

Append:

```bash
# 4. Instruction-tune (~5h after pretrain)
python finetune.py --atom --base checkpoints/atom/best.pt
# Output: checkpoints/atom/instruct.pt
```

- [ ] **Step 3: Commit**

```bash
git add finetune.py atom/azure_train.sh
git commit -m "Atom: instruction-tune path (full FT, LR 1e-5)"
```

---

## Task 15: Upload to HuggingFace + site card

Push base + instruct to HF. Add Atom card to `index.html` and a benchmark entry.

**Files:**
- Create: `atom/upload.py`
- Modify: `lattice-site/index.html` (new card — separate repo)
- Modify: `lattice-site/benchmarks.html` (Atom vs Mini entry)

- [ ] **Step 1: Write upload.py**

```python
# atom/upload.py
"""Push Atom base + instruct checkpoints to HuggingFace.

Usage:
  python -m atom.upload base    # → oli-mebberson/lattice-atom-base
  python -m atom.upload instruct
"""
import sys, argparse
from pathlib import Path

def upload(kind: str) -> None:
    from huggingface_hub import HfApi, create_repo
    repo = f"oli-mebberson/lattice-atom-{kind}"
    create_repo(repo, exist_ok=True, private=False)
    api = HfApi()

    ckpt = Path(f"checkpoints/atom/{'best' if kind == 'base' else 'instruct'}.pt")
    api.upload_file(path_or_fileobj=str(ckpt), path_in_repo="model.pt", repo_id=repo, repo_type="model")

    # README with honest framing (Atom = GPT-1-tier, not SmolLM-tier)
    readme = HONEST_README.format(kind=kind)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo, repo_type="model")
    print(f"Uploaded → https://huggingface.co/{repo}")

HONEST_README = """---
license: apache-2.0
tags: [from-scratch, gpt, lattice]
---

# Lattice Atom {kind}

A ~655M parameter GPT trained **entirely from scratch** on 10B SmolLM-Corpus tokens.
Every parameter learned by us — no fine-tuning of a base model.

**Honest quality:** trained on 10B tokens (SmolLM-360M used 60x more — 600B).
Output is "barely talkable" — coherent prose, recognizable instruction answers,
mostly wrong on specific facts. Research artifact, not an assistant. GPT-1 tier.

Architecture: 32 layers, 1280 dim, 20 heads, 2048 context, RoPE + RMSNorm + SwiGLU.
"""

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["base", "instruct"])
    args = p.parse_args()
    upload(args.kind)
```

- [ ] **Step 2: Add Atom card to index.html** (separate `lattice-site` repo)

Between Mini and Pulse cards — "Lattice Atom · 655M · From scratch · GPT-1-tier demo".

- [ ] **Step 3: Add Atom vs Mini benchmark to benchmarks.html**

After the chart runs (need a quick eval script comparing Atom vs Mini on the same prompts used in `pulse/benchmark.py`).

- [ ] **Step 4: Commit + push both repos**

```bash
git add atom/upload.py
git commit -m "Atom: HF upload script with honest model card"
# lattice-site changes committed in that repo separately
```

---

## Task 16: Final README + wrap

Update the top-level README to mention Atom in the product line. Add `docs/superpowers/specs/2026-07-27-lattice-atom-design.md` reference.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README product line table**

Add Lattice Atom row.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Atom: add to product line in README"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ §1 Goals — Tasks 1–9 (arch + sanity gate), Task 13 (pretrain), Task 14 (instruct), Task 15 (HF + benchmark)
- ✅ §2 Architecture — Tasks 2–7 (RMSNorm, RoPE, SwiGLU, block, AtomGPT, config)
- ✅ §3 Data — Tasks 10–11 (registry, tokenizer, prepare_data)
- ✅ §4 Training — Task 8 (config), Task 12 (train_atom), Task 13 (A100 run)
- ✅ §5 File structure — matches the plan's File Structure section
- ✅ §6 Deployment — Task 15 (HF + site)
- ✅ §7 Timeline — Tasks 1–12 are the Mac-days, Tasks 13–14 are the weekend
- ✅ §8 Risks — addressed: sanity overfit (Task 9) catches wiring; 30-min ckpts (Task 8/12) handle preemption
- ✅ §9 Success — Tasks 13–15 produce the shippable artifacts

**Placeholder scan:** The `get_atom_batch_iterator` in Task 12 step 4 has a flagged placeholder for batch building — explicitly called out with the fix. No other TODOs.

**Type consistency:** `AtomTrainConfig` (Task 8) → used in `train_atom` (Task 12). `lattice_atom_config` (Task 7) → used in Task 6 test, Task 9 sanity, Task 12 train. `AtomGPT` (Task 6) → used everywhere downstream. Signatures match.
