"""Load OpenAI's GPT-2 124M weights into the model from model.py.

The weights come from the Hugging Face repository openai-community/gpt2
(model.safetensors, about 548 MB, float32), downloaded once into data/.
"""
import torch
from safetensors.torch import load_file

from .data import download
from .model import GPT, GPTConfig

WEIGHTS_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"


def _copy(dst, src):
    if dst.shape != src.shape:
        raise ValueError(f"shape mismatch: model {tuple(dst.shape)} vs file {tuple(src.shape)}")
    with torch.no_grad():
        dst.copy_(src)


def load_gpt2(device="cpu", dropout=0.0):
    sd = load_file(download(WEIGHTS_URL, "gpt2-124M.safetensors"))
    model = GPT(GPTConfig(dropout=dropout, qkv_bias=True, tie_weights=True))

    _copy(model.tok_emb.weight, sd["wte.weight"])
    _copy(model.pos_emb.weight, sd["wpe.weight"])
    for i, block in enumerate(model.blocks):
        p = f"h.{i}."
        # OpenAI stored its linear layers as (in, out); nn.Linear keeps (out, in), hence .T
        _copy(block.attn.qkv.weight, sd[p + "attn.c_attn.weight"].T)
        _copy(block.attn.qkv.bias, sd[p + "attn.c_attn.bias"])
        _copy(block.attn.proj.weight, sd[p + "attn.c_proj.weight"].T)
        _copy(block.attn.proj.bias, sd[p + "attn.c_proj.bias"])
        _copy(block.norm1.weight, sd[p + "ln_1.weight"])
        _copy(block.norm1.bias, sd[p + "ln_1.bias"])
        _copy(block.norm2.weight, sd[p + "ln_2.weight"])
        _copy(block.norm2.bias, sd[p + "ln_2.bias"])
        _copy(block.ffn.fc.weight, sd[p + "mlp.c_fc.weight"].T)
        _copy(block.ffn.fc.bias, sd[p + "mlp.c_fc.bias"])
        _copy(block.ffn.proj.weight, sd[p + "mlp.c_proj.weight"].T)
        _copy(block.ffn.proj.bias, sd[p + "mlp.c_proj.bias"])
    _copy(model.final_norm.weight, sd["ln_f.weight"])
    _copy(model.final_norm.bias, sd["ln_f.bias"])
    # GPT-2 has no separate output matrix: the head reuses the token embedding (tie_weights=True).
    return model.to(device).eval()
