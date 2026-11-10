"""Chapter 6: load OpenAI's GPT-2 124M weights into our model and generate.

    python scripts/06_gpt2.py "The meaning of life is"
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from llm.data import pick_device
from llm.gpt2 import load_gpt2
from llm.model import count_params, generate
from llm.tokenizer import decode, encode

prompt = sys.argv[1] if len(sys.argv) > 1 else "Every effort moves you"
device = pick_device()
model = load_gpt2(device)
print(f"{count_params(model):,} parameters on {device}")
idx = torch.tensor([encode(prompt)], device=device)
print("[greedy]", decode(generate(model, idx, 40)[0].tolist()))
g = torch.Generator(device=device).manual_seed(123)
print("[top-k 40, temperature 0.8]", decode(generate(model, idx, 40, temperature=0.8, top_k=40, generator=g)[0].tolist()))

try:  # optional check against the Hugging Face implementation
    from transformers import GPT2LMHeadModel
    ref = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    x = torch.tensor([encode("The quick brown fox jumps over the lazy dog.")])
    with torch.no_grad():
        diff = (model.cpu()(x) - ref(x).logits).abs().max().item()
    print(f"largest logit difference from transformers: {diff:.2e}")
except ImportError:
    print("(install transformers to compare logits against the reference implementation)")
