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
