"""Downloads, and turning a long text into (input, target) training pairs."""
import os
import urllib.request

import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

TIME_MACHINE_URL = "https://www.gutenberg.org/cache/epub/35/pg35.txt"   # H. G. Wells, public domain


def download(url, name):
    """Fetch url into data/name once, and return the local path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "build-an-llm"})
        with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(path + ".part", path)
    return path


def load_book(path=None):
    """Read a Project Gutenberg text and keep only the book between its START and END markers."""
    path = path or download(TIME_MACHINE_URL, "pg35.txt")
    text = open(path, encoding="utf-8").read()
    start = text.index("\n", text.index("*** START OF")) + 1
    end = text.index("*** END OF")
    return text[start:end].strip()


class TokenWindows(Dataset):
    """A window of context_length tokens slides along the text, stride tokens at a time.

    Each window is an input; the same window shifted one token right is its target,
    so target[t] is the token that follows input[t].
    """

    def __init__(self, ids, context_length, stride):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.T = context_length
        self.starts = list(range(0, len(ids) - context_length, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return self.ids[s:s + self.T], self.ids[s + 1:s + self.T + 1]


def make_loader(ids, context_length, batch_size, shuffle, seed=0, stride=None):
    ds = TokenWindows(ids, context_length, stride or context_length)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, generator=g)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
