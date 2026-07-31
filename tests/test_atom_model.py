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

