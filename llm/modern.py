"""The GPT from model.py with each post-GPT-2 change behind a switch:
RMSNorm, rotary position embeddings (RoPE), SwiGLU and grouped-query attention."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GELU, LayerNorm

@dataclass
class Cfg:
    vocab_size: int = 50257
    context_length: int = 128
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 8          # < n_heads: grouped-query attention
    norm: str = "layer"          # "layer" | "rms"
    pos: str = "learned"         # "learned" | "rope"
    ffn: str = "gelu"            # "gelu" | "swiglu"
    bias: bool = True
    rope_base: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms

def rope_angles(head_dim, n_pos, base=10000.0):
    i = torch.arange(head_dim // 2)
    theta = base ** (-2 * i / head_dim)                 # one frequency per pair of dimensions
    ang = torch.arange(n_pos)[:, None] * theta[None, :]  # (positions, head_dim/2)
    return ang.cos(), ang.sin()

def rope(x, cos, sin):
    """Rotate each (even, odd) pair of x's last dimension by its position's angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.kv, self.hd = c.n_heads, c.n_kv_heads, c.d_model // c.n_heads
        self.W_q = nn.Linear(c.d_model, self.h * self.hd, bias=c.bias)
        self.W_k = nn.Linear(c.d_model, self.kv * self.hd, bias=c.bias)
        self.W_v = nn.Linear(c.d_model, self.kv * self.hd, bias=c.bias)
        self.out_proj = nn.Linear(c.d_model, c.d_model, bias=c.bias)
        self.use_rope = c.pos == "rope"

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q = self.W_q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        if self.use_rope:
            q, k = rope(q, cos[:T], sin[:T]), rope(k, cos[:T], sin[:T])
        if self.kv < self.h:                       # each group of query heads shares one k, v
            k = k.repeat_interleave(self.h // self.kv, dim=1)
            v = v.repeat_interleave(self.h // self.kv, dim=1)
        scores = q @ k.transpose(-2, -1) / self.hd ** 0.5
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        w = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        return self.out_proj((w @ v).transpose(1, 2).reshape(B, T, D))

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class MLP(nn.Module):
    def __init__(self, d, bias):
        super().__init__()
        self.fc, self.act, self.proj = nn.Linear(d, 4 * d, bias=bias), GELU(), nn.Linear(4 * d, d, bias=bias)

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        Norm = RMSNorm if c.norm == "rms" else LayerNorm
        self.norm1, self.norm2 = Norm(c.d_model), Norm(c.d_model)
        self.attn = Attention(c)
        self.ffn = SwiGLU(c.d_model, int(2 / 3 * 4 * c.d_model)) if c.ffn == "swiglu" else MLP(c.d_model, c.bias)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))

class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_emb = nn.Embedding(c.context_length, c.d_model) if c.pos == "learned" else None
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final_norm = RMSNorm(c.d_model) if c.norm == "rms" else LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        cos, sin = rope_angles(c.d_model // c.n_heads, c.context_length, c.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return self.head(self.final_norm(x))
