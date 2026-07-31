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

