"""Next-token training: the loss, an evaluation pass, and a plain AdamW loop."""
import time

import torch
import torch.nn.functional as F


def batch_loss(model, x, y):
    """Average cross-entropy of predicting y[t] from x[:t+1], over every position."""
    logits = model(x)                                    # (B, T, V)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())


@torch.no_grad()
def eval_loss(model, loader, device, max_batches=None):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        total += batch_loss(model, x.to(device), y.to(device)).item() * len(x)
        n += len(x)
    model.train()
    return total / n


def train(model, train_loader, val_loader, device, epochs, lr=4e-4, weight_decay=0.1, eval_every=20):
    """Returns a list of (step, train_loss, val_loss)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, step, t0 = [], 0, time.time()
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            loss = batch_loss(model, x.to(device), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0:
                tr = eval_loss(model, train_loader, device, max_batches=20)
                va = eval_loss(model, val_loader, device)
                history.append((step, tr, va))
                print(f"epoch {epoch + 1:3d}  step {step:5d}  train {tr:.3f}  val {va:.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return history
