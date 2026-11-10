"""Chapter 8: instruction tuning GPT-2 on a slice of databricks-dolly-15k (CC BY-SA 3.0).

    python scripts/08_instructions.py              # full fine-tune
    python scripts/08_instructions.py --lora 8     # LoRA, rank 8
"""
import argparse, json, os, random, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from llm.data import download, pick_device
from llm.gpt2 import load_gpt2
from llm.lora import add_lora
from llm.model import generate
from llm.tokenizer import EOT_ID, decode, encode

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--lora", type=int, default=0, help="LoRA rank; 0 trains every weight")
ap.add_argument("--n-train", type=int, default=1000)
ap.add_argument("--no-mask", action="store_true", help="also train on the prompt tokens")
args = ap.parse_args()
device = pick_device()

def prompt_of(r):
    p = f"### Instruction:\n{r['instruction'].strip()}\n\n"
    if r["context"].strip():
        p += f"### Input:\n{r['context'].strip()}\n\n"
    return p + "### Response:\n"

path = download("https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl", "dolly.jsonl")
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
rows = [r for r in rows if r["response"].strip() and len(encode(prompt_of(r) + r["response"])) < 256]
random.Random(123).shuffle(rows)
train, test = rows[:args.n_train], rows[args.n_train:args.n_train + 5]

def collate(batch):
    """Inputs and next-token targets; the prompt's targets are -100 so the loss skips them."""
    xs, ys = [], []
    for r in batch:
        p, a = encode(prompt_of(r)), encode(r["response"].strip()) + [EOT_ID]
        ids = p + a
        y = ids[1:]
        if not args.no_mask:
            y = [-100] * (len(p) - 1) + y[len(p) - 1:]
        xs.append(ids[:-1]); ys.append(y)
    T = max(map(len, xs))
    X = torch.full((len(xs), T), EOT_ID); Y = torch.full((len(xs), T), -100)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :len(x)] = torch.tensor(x); Y[i, :len(y)] = torch.tensor(y)
    return X.to(device), Y.to(device)

def answer(model, r):
    idx = torch.tensor([encode(prompt_of(r))], device=device)
    out = generate(model, idx, 80, stop_token=EOT_ID)[0, idx.shape[1]:].tolist()
    return decode([t for t in out if t != EOT_ID]).strip()

torch.manual_seed(123)
model = load_gpt2("cpu")
if args.lora:
    add_lora(model, rank=args.lora, alpha=2 * args.lora)
model.to(device)
print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
before = [answer(model, r) for r in test]

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.1)
gen = torch.Generator().manual_seed(123)
for epoch in range(args.epochs):
    model.train()
    order = torch.randperm(len(train), generator=gen).tolist()
    for step, i in enumerate(range(0, len(order), 4), 1):
        X, Y = collate([train[j] for j in order[i:i + 4]])
        loss = F.cross_entropy(model(X).flatten(0, 1), Y.flatten(), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"epoch {epoch + 1} step {step} loss {loss.item():.3f}", flush=True)

for r, b in zip(test, before):
    print("\n" + prompt_of(r).strip())
    print("  before:", b[:200].replace("\n", " "))
    print("  after: ", answer(model, r)[:200].replace("\n", " "))
