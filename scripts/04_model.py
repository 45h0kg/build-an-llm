"""Chapter 4: the full GPT-2 124M-shaped model, its parameter count, and an untrained sample."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from llm.model import GPT, GPTConfig, count_params, generate
from llm.tokenizer import decode, encode

torch.manual_seed(123)
untied = GPT(GPTConfig())
tied = GPT(GPTConfig(tie_weights=True))
print(f"parameters, separate output head: {count_params(untied):,}")
print(f"parameters, head tied to the token embedding: {count_params(tied):,}  (GPT-2 124M)")

idx = torch.tensor([encode("The Time Traveller")])
out = generate(untied, idx, max_new_tokens=15)
print("untrained model writes:", repr(decode(out[0].tolist())))
