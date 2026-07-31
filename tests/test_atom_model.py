"""Unit tests for Atom architecture components."""
import torch


def test_rmsnorm_shape_preserved():
    """RMSNorm output has the same shape as input."""
    from model import RMSNorm
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_normalizes():
    """Output has roughly unit RMS before the learned scale (weight=1)."""
    from model import RMSNorm
    norm = RMSNorm(64)
    norm.weight.data.fill_(1.0)  # neutral scale
    x = torch.randn(2, 1000, 64) * 5.0  # large variance input
    out = norm(x)
    rms = out.pow(2).mean(dim=-1, keepdim=True).sqrt()
    assert (rms - 1.0).abs().mean() < 0.01


def test_rope_attention_shape():
    """RoPE attention produces same shape as input."""
    from config import ModelConfig
    from model import RoPEAttention
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_embd=64, n_head=4, block_size=32)
    attn = RoPEAttention(cfg)
    x = torch.randn(2, 16, 64)
    out = attn(x)
    assert out.shape == (2, 16, 64)


def test_rope_deterministic_same_input():
    """Same input twice → same output (no randomness in eval mode)."""
    from config import ModelConfig
    from model import RoPEAttention
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_embd=64, n_head=4, block_size=32)
    torch.manual_seed(0)
    attn = RoPEAttention(cfg)
    attn.eval()
    x = torch.randn(1, 8, 64)
    out1 = attn(x)
    out2 = attn(x)
    assert torch.allclose(out1, out2)


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
    """Hidden dim is the multiple of 64 at/above 8/3*n_embd."""
    from config import ModelConfig
    from model import SwiGLU
    # 1280: 8/3*1280 = 3413.33, rounded up to multiple of 64 = 3456
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_embd=1280, n_head=20, block_size=2048)
    ff = SwiGLU(cfg)
    assert ff.hidden == 3456


def test_atom_block_residual():
    """Block output shape matches input (residual blocks preserve shape)."""
    from config import ModelConfig
    from model import AtomTransformerBlock
    cfg = ModelConfig(vocab_size=100, n_layer=2, n_embd=64, n_head=4, block_size=32)
    block = AtomTransformerBlock(cfg)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == (2, 16, 64)


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


def test_atom_gpt_generate():
    """AtomGPT can autoregressively generate tokens."""
    from config import ModelConfig
    from model import AtomGPT
    cfg = ModelConfig(vocab_size=256, n_layer=2, n_embd=64, n_head=4, block_size=32)
    model = AtomGPT(cfg)
    x = torch.randint(0, 256, (1, 4))
    out = model.generate(x, max_new_tokens=8, temperature=1.0, top_k=10)
    assert out.shape == (1, 12)  # 4 input + 8 new


def test_atom_gpt_param_count():
    """Full Atom config produces ~655M params (within 5%)."""
    from config import lattice_atom_config
    from model import AtomGPT
    model = AtomGPT(lattice_atom_config())
    n = sum(p.numel() for p in model.parameters())
    # Target 655M; allow ±5% for tied-embedding / bias variance
    assert 620_000_000 < n < 690_000_000, f"got {n:,} params"


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


def test_atom_train_config_defaults():
    """AtomTrainConfig has bf16 enabled, 30-min checkpoints, cosine LR."""
    from config import AtomTrainConfig
    cfg = AtomTrainConfig()
    assert cfg.use_bf16 is True
    assert cfg.checkpoint_minutes == 30
    assert cfg.batch_size * cfg.grad_accum_steps == 512  # effective batch
    assert cfg.learning_rate == 3e-4
    assert 0 < cfg.warmup_iters < cfg.lr_decay_iters
    assert cfg.max_iters == cfg.lr_decay_iters  # decay over full run

