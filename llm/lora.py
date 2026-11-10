"""LoRA: freeze a Linear layer and learn a small low-rank correction B @ A beside it."""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))   # zero, so it starts as a no-op
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(model, rank=8, alpha=16):
    """Freeze the whole model, then wrap every attention and feed-forward Linear with LoRA."""
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        a, f = block.attn, block.ffn
        a.qkv, a.proj = LoRALinear(a.qkv, rank, alpha), LoRALinear(a.proj, rank, alpha)
        f.fc, f.proj = LoRALinear(f.fc, rank, alpha), LoRALinear(f.proj, rank, alpha)
    return model
