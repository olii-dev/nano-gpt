"""
GPT-style decoder-only transformer, implemented from scratch in PyTorch.

Architecture overview (bottom → top):
  1. Token embedding  — maps each token ID to a learned vector
  2. Position embedding — adds "where in the sequence" information
  3. N × TransformerBlock:
       a. LayerNorm → Multi-Head Causal Self-Attention → residual add
       b. LayerNorm → Feed-Forward Network (MLP) → residual add
  4. Final LayerNorm → linear head → logits over vocabulary

This is the same family as GPT-2/3: decoder-only, causal (left-to-right),
pre-norm (LayerNorm before each sub-layer for training stability).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, count_parameters, model_config


# ---------------------------------------------------------------------------
# Causal self-attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (triangular) mask.

    Intuition: each token "asks" every previous token (including itself)
    "how relevant are you to me?" via dot-product scores.  The mask prevents
    peeking at future tokens during training — essential for autoregressive LM.

    Q, K, V are learned linear projections of the input, split into
    `n_head` independent heads so the model can attend to different
    relationships in parallel (syntax, semantics, position, etc.).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        # Single fused projection for Q, K, V (more efficient than 3 separate layers)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: upper triangle = -inf so softmax gives 0 attention to future
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Project to Q, K, V and reshape into heads
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # CUDA: fused SDPA kernel (faster, uses flash/mem-efficient backends when available)
        # MPS/CPU: explicit matmul — SDPA has been flaky on MPS; manual path is portable
        if q.device.type == "cuda":
            dropout_p = self.attn_dropout.p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, hd)

        # Merge heads back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# Feed-forward block (MLP)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise two-layer MLP applied identically to every token.

    Expands embedding dim 4× (standard GPT ratio), applies GELU non-linearity
    (smoother than ReLU — helps gradient flow), then projects back down.
    This is where the model stores most of its "knowledge" per layer.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One transformer layer: pre-norm attention + pre-norm FFN, each with
    a residual (skip) connection.

    Residual connections let gradients flow directly through the network,
    which is critical for training deep models (12+ layers).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Full language model
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class GPT(nn.Module):
    """
    Complete decoder-only language model.

    Weight tying: the token embedding matrix and the final lm_head share
    the same weights.  This cuts parameters roughly in half for the largest
  matrices and is standard in GPT-2+.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),   # token embed
            "wpe": nn.Embedding(config.block_size, config.n_embd),   # position embed
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

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
        """
        Args:
            idx: (B, T) token IDs
            targets: (B, T) next-token targets (optional — if given, compute loss)

        Returns:
            ModelOutput with logits (B, T, vocab) and optional scalar loss
        """
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy over vocabulary for every position
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
        """
        Autoregressive sampling: repeatedly append one token at a time.

        MPS workaround: move logits to CPU for multinomial / top-k / top-p
        (known MPS bugs).  CUDA and CPU keep sampling on the active device.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond).logits
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # MPS-only: sampling ops are unreliable on Metal — use CPU
            if idx.device.type == "mps":
                logits_sample = logits.float().cpu()
            else:
                logits_sample = logits.float()

            if top_k is not None:
                v, _ = torch.topk(logits_sample, min(top_k, logits_sample.size(-1)))
                logits_sample[logits_sample < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                logits_sample = _top_p_filter(logits_sample, top_p)

            probs = F.softmax(logits_sample, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.device != idx.device:
                next_id = next_id.to(idx.device)

            idx = torch.cat([idx, next_id], dim=1)

        return idx


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep smallest set of tokens whose cumulative prob >= top_p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[mask] = float("-inf")
    logits.scatter_(1, sorted_idx, sorted_logits)
    return logits


def build_model(config: ModelConfig | None = None) -> GPT:
    """Factory helper — builds model and prints parameter count."""
    config = config or model_config
    model = GPT(config)
    n_params = count_parameters(model)
    print(f"GPT model: {config.n_layer} layers, {config.n_embd} dim, "
          f"{config.n_head} heads, {config.block_size} context")
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    return model


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import get_device

    device = get_device()
    print(f"Device: {device}")

    # Tiny config for a fast smoke test
    cfg = ModelConfig(vocab_size=256, n_layer=2, n_embd=64, n_head=4, block_size=32)
    model = build_model(cfg).to(device)

    x = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    out = model(x, y)
    print(f"Logits shape: {out.logits.shape},  loss: {out.loss.item():.4f}")

    generated = model.generate(x[:1], max_new_tokens=10, temperature=0.8, top_k=10)
    print(f"Generated shape: {generated.shape}")
