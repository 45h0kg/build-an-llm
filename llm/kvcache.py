"""Generation with a KV cache: keep each block's keys and values so every new
token runs through the model alone instead of re-running the whole sequence."""
import math

import torch


def _attn_step(attn, x, cache):
    B, T, C = x.shape
    q, k, v = (attn.split_heads(t, B, T) for t in attn.qkv(x).split(C, dim=-1))
    if cache is not None:                        # append the new keys and values to the stored ones
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
    start = k.shape[2] - T                       # positions that came before x
    scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
    scores = scores.masked_fill(attn.mask[start:start + T, :start + T], float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    return attn.proj(out.transpose(1, 2).reshape(B, T, C)), (k, v)


def _forward_cached(model, idx, caches):
    start = 0 if caches[0] is None else caches[0][0].shape[2]
    pos = torch.arange(start, start + idx.shape[1], device=idx.device)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    new = []
    for block, cache in zip(model.blocks, caches):
        a, kv = _attn_step(block.attn, block.norm1(x), cache)
        x = x + a
        x = x + block.ffn(block.norm2(x))
        new.append(kv)
    return model.head(model.final_norm(x)), new


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens):
    """Greedy generation; produces the same tokens as model.generate(..., temperature=0)."""
    model.eval()
    assert idx.shape[1] + max_new_tokens <= model.cfg.context_length
    logits, caches = _forward_cached(model, idx, [None] * len(model.blocks))   # the prompt once
    for _ in range(max_new_tokens):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        logits, caches = _forward_cached(model, nxt, caches)                    # then one token at a time
    return idx
