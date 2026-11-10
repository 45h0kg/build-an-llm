"""Chapter 1: text to numbers. A tiny BPE learned by hand, then GPT-2's real one."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.data import load_book
from llm.tokenizer import bpe_encode_word, encode, decode, show_tokens, train_bpe

corpus = "low low low low lower lower newest newest newest widest widest"
print("== toy BPE on:", corpus)
merges = train_bpe(corpus, 6, verbose=True)
print("'lowest' ->", bpe_encode_word("lowest", merges))

print("\n== GPT-2's BPE (vocabulary 50,257)")
for text in ["The Time Traveller smiled.", "Tokenization isn't word splitting", " hello hello"]:
    print(repr(text), "->", show_tokens(text))
ids = encode("Hello, world")
print("round trip:", ids, "->", repr(decode(ids)))

book = load_book()
print(f"\nThe Time Machine: {len(book):,} characters, {len(encode(book)):,} GPT-2 tokens")
