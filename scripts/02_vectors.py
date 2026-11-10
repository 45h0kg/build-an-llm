"""Chapter 2: token IDs become vectors, and a long text becomes (input, target) batches."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from llm.data import load_book, make_loader
from llm.tokenizer import decode, encode

torch.manual_seed(123)
vocab, dim = 6, 3
table = torch.nn.Embedding(vocab, dim)
ids = torch.tensor([2, 5, 2])
one_hot = torch.nn.functional.one_hot(ids, vocab).float()
print("embedding lookup:\n", table(ids))
print("one-hot @ weight matrix gives the same rows:", torch.allclose(table(ids), one_hot @ table.weight))

# Without positions, the same token gets the same vector wherever it appears.
pos = torch.nn.Embedding(4, dim)
x = table(ids) + pos(torch.arange(3))
print("token 2 at positions 0 and 2, after adding position vectors, differ:", not torch.allclose(x[0], x[2]))

loader = make_loader(encode(load_book()), context_length=8, batch_size=4, shuffle=False, stride=8)
x, y = next(iter(loader))
print("\nbatch shapes:", tuple(x.shape), tuple(y.shape))
print("input :", repr(decode(x[0].tolist())))
print("target:", repr(decode(y[0].tolist())), "(the same window, one token later)")
