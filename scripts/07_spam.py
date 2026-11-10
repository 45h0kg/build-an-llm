"""Chapter 7: turn GPT-2 into a spam filter by swapping its output head for a two-way one.

    python scripts/07_spam.py --train last      # head | last | all
Data: UCI SMS Spam Collection (CC BY 4.0).
"""
import argparse, os, random, sys, zipfile; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from llm.data import DATA_DIR, download, pick_device
from llm.gpt2 import load_gpt2
from llm.tokenizer import EOT_ID, encode

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="last", choices=["head", "last", "all"])
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=5e-5)
args = ap.parse_args()
device = pick_device()

def load_splits(seed=123):
    z = download("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip", "sms.zip")
    lines = zipfile.ZipFile(z).read("SMSSpamCollection").decode("utf-8").splitlines()
    rows, seen = [], set()
    for line in lines:
        label, text = line.split("\t", 1)
        if text not in seen:                       # the file repeats some messages verbatim
            seen.add(text)
            rows.append((text, int(label == "spam")))
    spam = [r for r in rows if r[1]]
    rng = random.Random(seed)
    data = spam + rng.sample([r for r in rows if not r[1]], len(spam))   # balance the classes
    rng.shuffle(data)
    a, b = int(0.7 * len(data)), int(0.8 * len(data))
    return data[:a], data[a:b], data[b:]

def batches(rows, bs, gen=None):
    order = torch.randperm(len(rows), generator=gen).tolist() if gen else range(len(rows))
    order = list(order)
    for i in range(0, len(rows), bs):
        chunk = [rows[j] for j in order[i:i + bs]]
        ids = [encode(t)[:128] for t, _ in chunk]
        x = torch.full((len(ids), max(map(len, ids))), EOT_ID)
        for r, s in enumerate(ids):
            x[r, :len(s)] = torch.tensor(s)
        last = torch.tensor([len(s) - 1 for s in ids])
        yield x.to(device), last.to(device), torch.tensor([y for _, y in chunk], device=device)

def class_logits(model, x, last):
    return model(x)[torch.arange(len(x), device=device), last]   # the prediction at each message's last token

@torch.no_grad()
def accuracy(model, rows):
    model.eval()
    right = sum((class_logits(model, x, l).argmax(-1) == y).sum().item() for x, l, y in batches(rows, 32))
    model.train()
    return right / len(rows)

train, val, test = load_splits()
torch.manual_seed(123)
model = load_gpt2("cpu")
model.head = nn.Linear(768, 2)                 # two outputs: ham, spam
for p in model.parameters():
    p.requires_grad = False
parts = {"head": [model.head], "last": [model.head, model.final_norm, model.blocks[-1]], "all": [model]}[args.train]
for m in parts:
    for p in m.parameters():
        p.requires_grad = True
model.to(device)
print(f"{len(train)} train / {len(val)} val / {len(test)} test messages; training '{args.train}' on {device}")
print(f"test accuracy before: {accuracy(model, test):.1%}")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.1)
gen = torch.Generator().manual_seed(123)
for epoch in range(args.epochs):
    for x, last, y in batches(train, 8, gen):
        loss = F.cross_entropy(class_logits(model, x, last), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch + 1}: train {accuracy(model, train):.1%}  val {accuracy(model, val):.1%}")
print(f"test accuracy after: {accuracy(model, test):.1%}")

for msg in ["Are we still on for dinner tonight?", "WINNER! Claim your free prize now, text WIN to 80086"]:
    x, last, _ = next(batches([(msg, 0)], 1))
    model.eval()
    with torch.no_grad():
        p = torch.softmax(class_logits(model, x, last), -1)[0, 1].item()
    print(f"P(spam) = {p:.3f}  {msg!r}")
