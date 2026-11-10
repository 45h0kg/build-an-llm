"""Tokenization: a tiny byte-pair-encoding trainer, and GPT-2's real BPE via tiktoken."""
from collections import Counter

import tiktoken

# ---------------------------------------------------------------- toy BPE


def pair_counts(words):
    """Count adjacent symbol pairs. `words` maps a tuple of symbols to how often it occurs."""
    counts = Counter()
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts


def merge_pair(words, pair):
    """Replace every adjacent occurrence of `pair` with one joined symbol."""
    joined = pair[0] + pair[1]
    out = {}
    for symbols, freq in words.items():
        new, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                new.append(joined)
                i += 2
            else:
                new.append(symbols[i])
                i += 1
        out[tuple(new)] = out.get(tuple(new), 0) + freq
    return out


def train_bpe(text, num_merges, verbose=False):
    """Learn `num_merges` merges from the whitespace-separated words in `text`.

    Ties go to the pair seen first, so runs are deterministic.
    """
    words = Counter(tuple(w) for w in text.split())
    merges = []
    for step in range(1, num_merges + 1):
        counts = pair_counts(words)
        if not counts:
            break
        best = max(counts, key=counts.get)
        merges.append(best)
        words = merge_pair(words, best)
        if verbose:
            shown = "  ".join(" ".join(s) + (f" (x{f})" if f > 1 else "") for s, f in words.items())
            print(f"merge {step}: {best[0]!r} + {best[1]!r} -> {best[0] + best[1]!r}  seen {counts[best]} times")
            print("   ", shown)
    return merges


def bpe_encode_word(word, merges):
    """Split one word into symbols by replaying the merges in the order they were learned."""
    symbols = tuple(word)
    for pair in merges:
        symbols = next(iter(merge_pair({symbols: 1}, pair)))
    return list(symbols)


# ---------------------------------------------------------------- GPT-2 BPE

gpt2 = tiktoken.get_encoding("gpt2")
END_OF_TEXT = "<|endoftext|>"
EOT_ID = 50256


def encode(text):
    """Text -> list of GPT-2 token IDs. <|endoftext|> is allowed as a special token."""
    return gpt2.encode(text, allowed_special={END_OF_TEXT})


def decode(ids):
    """List of GPT-2 token IDs -> text."""
    return gpt2.decode(list(ids))


def show_tokens(text):
    """The pieces GPT-2 cuts `text` into, as (id, text) pairs."""
    return [(i, gpt2.decode_single_token_bytes(i).decode("utf-8", errors="replace")) for i in encode(text)]
