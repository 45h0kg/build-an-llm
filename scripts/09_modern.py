"""Chapter 9: GPT-2's design against the changes newer models made, trained briefly on the same book.

    python scripts/09_modern.py --steps 600
"""
import argparse, os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from llm.data import load_book, make_loader, pick_device
from llm.modern import Cfg, Model
from llm.tokenizer import encode
from llm.train import batch_loss, eval_loss

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--device", default=pick_device())
args = ap.parse_args()

ids = encode(load_book())
split = int(0.9 * len(ids))
variants = {
    "GPT-2 style": {},
    "+ RMSNorm": {"norm": "rms"},
    "+ RoPE": {"pos": "rope"},
    "+ SwiGLU": {"ffn": "swiglu"},
    "+ GQA (2 KV heads)": {"n_kv_heads": 2},
    "all four, no biases": {"norm": "rms", "pos": "rope", "ffn": "swiglu", "n_kv_heads": 2, "bias": False},
}
for name, kw in variants.items():
    torch.manual_seed(1)
    cfg = Cfg(**kw)
    model = Model(cfg).to(args.device)
    tl = make_loader(ids[:split], cfg.context_length, 16, shuffle=True, seed=1)
    vl = make_loader(ids[split:], cfg.context_length, 16, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    step = 0
    while step < args.steps:
        for x, y in tl:
            loss = batch_loss(model, x.to(args.device), y.to(args.device))
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step == args.steps:
                break
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:22s} {params / 1e6:5.1f}M params  val loss {eval_loss(model, vl, args.device):.3f}", flush=True)
