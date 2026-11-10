"""Chapter 5: train a small GPT on The Time Machine, then sample from it several ways.

    python scripts/05_train.py --epochs 10
"""
import argparse, os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from llm.data import load_book, make_loader, pick_device
from llm.model import GPT, GPTConfig, count_params, generate
from llm.tokenizer import decode, encode
from llm.train import train

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--device", default=pick_device())
args = ap.parse_args()

torch.manual_seed(123)
ids = encode(load_book())
split = int(0.9 * len(ids))
cfg = GPTConfig(context_length=128, emb_dim=256, n_layers=4, n_heads=4, dropout=0.1)
train_loader = make_loader(ids[:split], cfg.context_length, batch_size=16, shuffle=True, seed=123)
val_loader = make_loader(ids[split:], cfg.context_length, batch_size=16, shuffle=False)
model = GPT(cfg).to(args.device)
print(f"{count_params(model):,} parameters, {len(ids):,} tokens, device {args.device}")

train(model, train_loader, val_loader, args.device, epochs=args.epochs)
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/small.pt")

prompt = torch.tensor([encode("The Time Traveller")], device=args.device)
g = torch.Generator(device=args.device).manual_seed(123)
for name, kw in [("greedy", {}), ("temperature 0.7", {"temperature": 0.7}),
                 ("temperature 1.5", {"temperature": 1.5}), ("top-k 20, temperature 1.0", {"temperature": 1.0, "top_k": 20})]:
    out = generate(model, prompt, 40, generator=g, **kw)
    print(f"\n[{name}]", decode(out[0].tolist()))
