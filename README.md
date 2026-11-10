# Build an LLM

A small GPT built piece by piece in plain PyTorch: tokenization, embeddings, causal attention, the transformer block, training, loading OpenAI's GPT-2 124M weights, fine-tuning for a task and for instructions, and the changes newer models made since GPT-2.

This is the code from the [Building an LLM](https://ashokg.ai/llm) track on ashokg.ai, cut down to the parts that matter. Each chapter there explains one piece and has a matching script here.

## Layout

```
llm/
  tokenizer.py   a tiny byte-pair-encoding trainer, and GPT-2's BPE via tiktoken
  data.py        downloads, sliding-window (input, target) batches
  model.py       the GPT: LayerNorm, GELU, causal attention, block, model, generate()
  train.py       next-token loss and a plain AdamW loop
  gpt2.py        loads OpenAI's GPT-2 124M weights into model.py
  lora.py        LoRA adapters
  kvcache.py     greedy generation with a KV cache
  modern.py      RMSNorm, RoPE, SwiGLU and grouped-query attention behind switches
scripts/
  01_tokens.py ... 09_modern.py, one per chapter
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer. Scripts pick CUDA, then Apple's MPS, then the CPU. Data and weights download into `data/` on first use.

## The scripts

| Script | Chapter | What it does |
|---|---|---|
| `01_tokens.py` | [From text to numbers](https://ashokg.ai/llm/01-from-text-to-numbers) | Learns a tiny BPE merge by merge, then shows GPT-2's tokens |
| `02_vectors.py` | [Tokens become vectors](https://ashokg.ai/llm/02-tokens-become-vectors) | Embedding lookup as one-hot × matrix, positions, batch shapes |
| `03_attention.py` | [Attention, slowly](https://ashokg.ai/llm/03-attention-slowly) | Causal attention on a 3×4 example, then the multi-head module |
| `04_model.py` | [The transformer block](https://ashokg.ai/llm/04-the-transformer-block) | Builds the GPT-2 124M shape, counts 124,439,808 parameters, samples from it untrained |
| `05_train.py` | [Learning to predict](https://ashokg.ai/llm/05-learning-to-predict) | Trains a 29M-parameter model on *The Time Machine*, then samples greedy, with temperature and top-k |
| `06_gpt2.py` | [Borrowing GPT-2's weights](https://ashokg.ai/llm/06-borrowing-gpt2s-weights) | Loads GPT-2 124M and generates; compares logits with `transformers` if installed |
| `07_spam.py` | [Teaching it one job](https://ashokg.ai/llm/07-teaching-it-one-job) | Swaps the output head for two classes and fine-tunes on SMS spam |
| `08_instructions.py` | [Teaching it to follow instructions](https://ashokg.ai/llm/08-teaching-it-to-follow-instructions) | Instruction tuning with the prompt masked out of the loss; `--lora 8` for LoRA |
| `09_kv_cache.py` | [What changed since GPT-2](https://ashokg.ai/llm/09-what-changed-since-gpt2) | Generation with and without a KV cache: same tokens, timed |
| `09_modern.py` | [What changed since GPT-2](https://ashokg.ai/llm/09-what-changed-since-gpt2) | Trains GPT-2-style and modern variants briefly on the same book |

```bash
python scripts/04_model.py
python scripts/05_train.py --epochs 30
python scripts/06_gpt2.py "The meaning of life is"
python scripts/07_spam.py --train last
python scripts/08_instructions.py --lora 8
```

The fine-tuning scripts (07, 08) train the full 124M model and are meant for a GPU or Apple silicon. `08_instructions.py` without LoRA needs about 10 GB of GPU memory at its batch size of 4.

## Data and weights

| What | Source | License |
|---|---|---|
| *The Time Machine*, H. G. Wells | [Project Gutenberg #35](https://www.gutenberg.org/ebooks/35) | Public domain in the US |
| GPT-2 124M weights | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | [Modified MIT](https://github.com/openai/gpt-2/blob/master/LICENSE) |
| SMS Spam Collection | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | CC BY 4.0 |
| databricks-dolly-15k | [Hugging Face](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | CC BY-SA 3.0 |

None of these files are stored in this repository.

## License

MIT, for the code in this repository.
