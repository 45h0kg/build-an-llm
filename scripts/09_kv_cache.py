"""Chapter 9: the KV cache. Same tokens, far less work per new token."""
import os, sys, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from llm.gpt2 import load_gpt2
from llm.kvcache import generate_cached
from llm.model import generate
from llm.tokenizer import encode

model = load_gpt2("cpu")
idx = torch.tensor([encode("The history of the bicycle begins")])
for n in [50, 200]:
    t = time.perf_counter(); a = generate(model, idx, n); t1 = time.perf_counter() - t
    t = time.perf_counter(); b = generate_cached(model, idx, n); t2 = time.perf_counter() - t
    print(f"{n} tokens: no cache {t1:.2f}s, cache {t2:.2f}s ({t1 / t2:.1f}x), identical: {torch.equal(a, b)}")
