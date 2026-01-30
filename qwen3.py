import os
import json
from typing import Optional, List, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from config import CausalLMOutput
from config import Qwen3Config

class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim=128, base=1000000):  # base from `config.rope_theta`
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # shape: [D/2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()    # shape: [T]            T: seq_len
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)   # shape: [T, D/2]
        freqs = freqs.unsqueeze(0).unsqueeze(1)             # shape: [1, 1, T, D/2]
        return freqs.cos(), freqs.sin()


def apply_rotary(x, cos, sin):
    """
    x:   [B, H, T, D] or [B, 2H, T, D]
    cos: [1, 1, T, D/2]
    sin: [1, 1, T, D/2]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)



class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(self.head_dim)
        self.k_norm = Qwen3RMSNorm(self.head_dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        # projections
        q = self.q_proj(x)   # [B, T, 2H*D]  2048 = 2 x num_heads(8) * head_dim(128)
        k = self.k_proj(x)   # [B, T, H*D]   1024 = num_heads(8) * head_dim(128)
        v = self.v_proj(x)   # (B, T, H*D)   H*D = 1024 = num_heads(8) * head_dim(128)

        # reshape -> [B, heads, T, D]
        q = q.view(B, T, 2 * H, D).transpose(1, 2)      # [B, 2H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2)          # [B, H, T, D]
        v = v.view(B, T, H, D).transpose(1, 2)          # [B, H, T, D]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE (sequence-aware)
        # q.shape:               [B, 2H, T, D]
        # k.shape:               [B, H, T, D]
        # cos.shape = sin.shape: [1, 1, T, D/2]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # --- GQA: map 2H Q heads -> H KV heads
        # q: [B, 2H, T, D]
        # The repeat_interleave is optinal, tensor will broadcast `H` to `2H` at dim 1.
        k = k.repeat_interleave(2, dim=1)  # [B, 2H, T, D]
        v = v.repeat_interleave(2, dim=1)  # [B, 2H, T, D]


        # --- Attention scores: token x token
        # [B, 2H, T, D] @ [B, 2H, D, T] -> [B, 2H, T, T]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, 2H, T, T]

        # causal mask on sequence
        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device)   # [T, T]
        ).bool()
        """Keep value on `True`
        causal_mask (T x T):
       [[ True, False, False, False, False],
        [ True,  True, False, False, False],
        [ True,  True,  True, False, False],
        [ True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True]]
        """

        attn = attn.masked_fill(~causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        """
        attn[0, 0, :, :] (T, T):
       [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.7173, 0.2827, 0.0000, 0.0000, 0.0000],
        [0.4697, 0.2938, 0.2365, 0.0000, 0.0000],
        [0.4081, 0.2351, 0.1630, 0.1937, 0.0000],
        [0.3014, 0.1474, 0.0836, 0.3023, 0.1652]]
        """

        # attention output
        # [B, 2H, T, T] @ [B, 2H, T, D] -> [B, 2H, T, D]
        out = torch.matmul(attn, v)

        # fold heads back
        # [B, 2H, T, D] -> [B, T, 2H, D] -> [B, T, 2 * H * D]
        out = out.transpose(1, 2).reshape(B, T, 2 * H * D)

        # [B, T, 2H * D] @ [2H*D, H*D] -> [B, T, H*D]
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=3072):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Qwen3Attention()
        self.mlp = Qwen3MLP()
        self.input_layernorm = Qwen3RMSNorm(1024)
        self.post_attention_layernorm = Qwen3RMSNorm(1024)

    def forward(self, x, cos, sin):
        # x.shape: [B, T, embed_dim]  embed_dim = HxD
        # after self_attn: [B, T, H*D]  [1, 5, 1024(8 x 128)]
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        # x.shape: [B, T, embed_dim]  # shape always the same
        return x


class Qwen3Model(nn.Module):
    def __init__(self, vocab_size=151936, num_layers=28):  # vocab_size(151936) from `config.vocab_size`
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, 1024)  # each token embedding dim: 1024
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer() for _ in range(num_layers)]
        )
        self.norm = Qwen3RMSNorm(1024)
        self.rotary_emb = Qwen3RotaryEmbedding(128)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        B, T, _ = x.shape
        # print(f"{B=}  {T=}  {x.shape=}")
        # B=1  T=5  x.shape=torch.Size([1, 5, 1024]) [B, T, embed_dim] [B, T, embed_dim]
        cos, sin = self.rotary_emb(T, x.device)
        # print(f"{cos.shape=}  {sin.shape=}  {x.shape=}")
        # cos.shape = sin.shape: [1, 1, T, D/2]  [1, 1, 5, 64]
        for layer in self.layers:
            x = layer(x, cos, sin)

        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Qwen3Model()
        self.lm_head = nn.Linear(1024, 151936, bias=False)  # vocab_size(151936) from `config.vocab_size`

    def forward(self, input_ids, return_dict: bool = True):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        if not return_dict:
            return logits
        return CausalLMOutput(logits=logits)

    @classmethod
    def _convert_state_dict(cls, state_dict: Dict):
        return state_dict

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        device="cpu",
        dtype=torch.float32,
        strict=False,
    ):
        """
        Load model + weights from a HuggingFace-style directory.
        """
        # ---- 1. load config ----
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config = Qwen3Config.from_dict(json.load(f))

        # ---- 2. init model ----
        model = cls()
        model.to(device=device, dtype=dtype)

        # ---- 3. load weights to memory ----
        weight_path = os.path.join(model_dir, "model.safetensors")
        state_dict = load_file(weight_path, device=device)

        # ---- 4. optional key fixups ----
        state_dict = cls._convert_state_dict(state_dict)

        # ---- 5. load weights to model ----
        missing, unexpected = model.load_state_dict(
            state_dict, strict=strict
        )

        if missing:
            print("⚠️ Missing keys:", missing)
        if unexpected:
            print("⚠️ Unexpected keys:", unexpected)

        model.eval()
        return model


if __name__ == "__main__":
    # ----------------- Simple check ----------------- #
    model = Qwen3ForCausalLM()
    model.eval()

    print(model)

    torch.manual_seed(0)
    x = torch.randint(0, 151936, (1, 8))   # x.shape: [1, 5]
    with torch.no_grad():
        logits = model(x).logits                 # [B, T, vocab_size] [1, 5, 151936]
        # If greedy, softmax is not necessary, since the index of max value keeps same.
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        print(f"prompt: {list(x)}\nnext_token: {next_token}")
        print("Logits[10:]", logits[0, -1, :10])

    # ----------------- Load weights ----------------- #
    # If don't have weights, comment later code
    ckpt_dir = "/data/weights/Qwen3-0.6B"
    model = Qwen3ForCausalLM.from_pretrained(ckpt_dir)
    with torch.no_grad():
        logits = model(x).logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        print(f"prompt: {list(x)}\nnext_token: {next_token}")
        print("Logits[10:]", logits[0, -1, :10])
