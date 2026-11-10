"""Chapter 3: attention on numbers small enough to check by hand, then the real module."""
import math, os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from llm.model import CausalSelfAttention

torch.set_printoptions(precision=3, sci_mode=False)
# Three tokens, four numbers each.
x = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                  [0.0, 2.0, 0.0, 2.0],
                  [1.0, 1.0, 1.0, 1.0]])
scores = x @ x.T / math.sqrt(x.shape[1])               # how similar each pair is, scaled by sqrt(d)
print("scores:\n", scores)
mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
print("causal attention weights (rows sum to 1, nothing above the diagonal):\n", weights)
print("outputs, each a weighted mix of the rows it may see:\n", weights @ x)

torch.manual_seed(123)
attn = CausalSelfAttention(emb_dim=768, n_heads=12, context_length=1024)
out = attn(torch.randn(2, 10, 768))
print("\nmulti-head module: (2, 10, 768) in ->", tuple(out.shape), "out")
