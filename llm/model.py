"""A GPT-2-shaped language model in plain PyTorch.

Learned token and position embeddings, a stack of pre-LayerNorm transformer
blocks (causal multi-head attention, then a 4x GELU feed-forward network, each
wrapped in a residual connection), a final LayerNorm and a linear output head.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 50257      # GPT-2 BPE vocabulary
    context_length: int = 1024   # longest sequence the position table covers
    emb_dim: int = 768           # width of every token vector
    n_layers: int = 12           # transformer blocks
    n_heads: int = 12            # attention heads per block
    dropout: float = 0.1         # used only while training
    qkv_bias: bool = True        # GPT-2 has biases on the q, k, v projections
    tie_weights: bool = False    # share the token embedding with the output head


class LayerNorm(nn.Module):
    """Normalize each vector to mean 0 and variance 1, then scale and shift."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class GELU(nn.Module):
    """GELU with the tanh approximation GPT-2 uses."""

    def forward(self, x):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))


class CausalSelfAttention(nn.Module):
    """Multi-head attention where each position sees only itself and earlier positions."""

    def __init__(self, emb_dim, n_heads, context_length, dropout=0.0, qkv_bias=True):
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must split evenly across heads"
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=qkv_bias)  # queries, keys, values in one matrix
        self.proj = nn.Linear(emb_dim, emb_dim)                      # mixes the heads back together
        self.attn_drop = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)        # True above the diagonal = the future

    def split_heads(self, t, B, T):
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # (B, heads, T, head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (self.split_heads(t, B, T) for t in self.qkv(x).split(C, dim=-1))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = self.attn_drop(weights) @ v
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class FeedForward(nn.Module):
    """Expand each vector 4x, apply GELU, project back. Each position on its own."""

    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 4 * emb_dim)
        self.act = GELU()
        self.proj = nn.Linear(4 * emb_dim, emb_dim)

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))


class Block(nn.Module):
    """One transformer block: x + attn(norm(x)), then x + ffn(norm(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.attn = CausalSelfAttention(cfg.emb_dim, cfg.n_heads, cfg.context_length,
                                        cfg.dropout, cfg.qkv_bias)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.ffn = FeedForward(cfg.emb_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        self.apply(self._init)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init(m):
        # Small normal weights (std 0.02, as in GPT-2) and zero biases.
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.cfg.context_length, "sequence longer than the context"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        return self.head(self.final_norm(x))   # (B, T, vocab_size) logits


def count_params(model):
    """Unique parameters; a tied weight counts once."""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.0, top_k=None, generator=None, stop_token=None):
    """Append up to max_new_tokens tokens to idx, shape (B, T).

    temperature=0 is greedy (always the most likely token). Otherwise the logits
    are divided by the temperature, optionally cut to the top_k largest, and sampled.
    """
    model.eval()
    ctx = model.cfg.context_length
    for _ in range(max_new_tokens):
        logits = model(idx[:, -ctx:])[:, -1, :]
        if temperature == 0.0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
        if stop_token is not None and (nxt == stop_token).all():
            break
    return idx
