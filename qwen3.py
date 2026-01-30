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

    def forward(self, x, cos, sin, past_kv=None):
        """
        x: [B, T_new, hidden]
        cos/sin: RoPE for total length (past_len + T_new)
        past_kv: tuple(k_cache, v_cache) where
                 k_cache: [B, 2H, past_len, D]
                 v_cache: [B, 2H, past_len, D]
        returns:
            out:    [B, T_new, hidden]
            new_kv: (k_total, v_total)

        `T_new` will be seq_len when prefill, and 1 when decode.
        For simplicity, use `T` to replace `T_new` in the code.
        """
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        # projections
        q = self.q_proj(x)   # [B, T, 2H*D]
        k = self.k_proj(x)   # [B, T, H*D]
        v = self.v_proj(x)   # (B, T, H*D)

        # reshape -> [B, heads, T, D]
        q = q.view(B, T, 2 * H, D).transpose(1, 2)      # [B, 2H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2)          # [B, H, T, D]
        v = v.view(B, T, H, D).transpose(1, 2)          # [B, H, T, D]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE (sequence-aware)
        # q.shape:               [B, 2H, T, D]
        # k.shape:               [B, H, T, D]
        # cos.shape = sin.shape: [1, 1, T_all_token, D/2]
        past_len = 0 if past_kv is None else past_kv[0].size(2)
        cos_new = cos[:, :, past_len:past_len + T, :]
        sin_new = sin[:, :, past_len:past_len + T, :]

        q = apply_rotary(q, cos_new, sin_new)
        k = apply_rotary(k, cos_new, sin_new)

        # --- append cache
        if past_kv is not None:
            k_cache, v_cache = past_kv
            k = torch.cat([k_cache, k], dim=2)  # [B, 2H, past_len+T, D]
            v = torch.cat([v_cache, v], dim=2)

        new_kv = (k, v)

        # attention over total length
        T_total = k.size(2)                # T_total = past_len+T

        # in dimension 1, `2H` and `H` will broadcast automatically
        # q: [B, 2H, T, D]
        k = k.repeat_interleave(2, dim=1)  # [B, 2H, T_total, D]
        v = v.repeat_interleave(2, dim=1)  # [B, 2H, T_total, D]

        # --- Attention scores: token x token
        # dim1 will broadcast `H` to `2H`
        # [B, 2H, T, D] @ [B, 2H, D, T_total] -> [B, 2H, T, T_total]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # causal mask for incremental decoding
        # query positions are [past_len ... past_len+T-1]
        # key positions are [0 ... T_total-1]
        q_pos = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(1)  # [T, 1]
        k_pos = torch.arange(T_total, device=x.device).unsqueeze(0)                 # [1, T_total]
        causal_mask = (k_pos <= q_pos)     # [T, T_total]
        """Keep value on `True`
        causal_mask (T x T_total):

        prefill:
        [[True,  False, False],
        [ True,  True,  False],
        [ True,  True,  True]]

        decode: all true
        [[True, True, True, True, True, True, True, True, True, True, True]]
        """

        attn = attn.masked_fill(~causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # attention output
        # [B, 2H, T, T_total] @ [B, 2H, T_total, D] -> [B, 2H, T, D]
        out = torch.matmul(attn, v)

        # fold heads back
        # [B, 2H, T, D] -> [B, T, 2H, D] -> [B, T, 2 * H * D]
        out = out.transpose(1, 2).reshape(B, T, 2 * H * D)

        # [B, T, 2H * D] @ [2H*D, H*D] -> [B, T, H*D]
        out = self.o_proj(out)

        return out, new_kv


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

    def forward(self, x, cos, sin, past_kv=None):
        """
        x:        [B, T_new, embed_dim]  embed_dim = HxD
        cos, sin: [T_total, D/2]         T_total = past_len + T_new
        """
        # after self_attn: [B, T, H*D]
        attn_out, new_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Qwen3Model(nn.Module):
    def __init__(self, vocab_size=151936, num_layers=28):  # vocab_size(151936) from `config.vocab_size`
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, 1024)  # each token embedding dim: 1024
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer() for _ in range(num_layers)]
        )
        self.norm = Qwen3RMSNorm(1024)
        self.rotary_emb = Qwen3RotaryEmbedding(128)

    def forward(self, input_ids, past_key_values, use_cache=False):
        """
        input_ids: [B, T_new]
        past_key_values: list of length num_layers, each is (k, v)
            [(k, v), (k, v)]

        For simplicity, use `T` to replace `T_new` in the code.
        """
        x = self.embed_tokens(input_ids)
        B, T, _ = x.shape

        # compute total length for RoPE
        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values[0][0].size(2)

        total_len = past_len + T
        # cos, sin: [total_len, D/2]
        cos, sin = self.rotary_emb(total_len, x.device)

        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            x, new_kv = layer(x, cos, sin, past_kv=layer_past)
            if use_cache:
                new_past_key_values.append(new_kv)

        x = self.norm(x)
        return x, new_past_key_values


class Qwen3ForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Qwen3Model()
        self.lm_head = nn.Linear(1024, 151936, bias=False)  # vocab_size(151936) from `config.vocab_size`

    def forward(self,
        input_ids,
        past_key_values=None,
        use_cache=False,
        return_dict: bool = True
    ):
        hidden_states, new_past = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        logits = self.lm_head(hidden_states)
        if not return_dict:
            return logits
        return CausalLMOutput(logits=logits, past_key_values=new_past)

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
    x = torch.tensor([[33464, 6832, 374]], dtype=torch.int32)

    past = None
    with torch.no_grad():
        # 1) prefill: run full prompt ONCE, cache everything
        out = model(x, past_key_values=None, use_cache=True)
        past = out.past_key_values

        # next token from prompt end
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        print("first next_token:", next_token.item())

        # 2) decode: decode one token at a time using cache
        for step in range(5):
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values

            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            print(f"step={step} next_token={next_token.item()}")

    # ----------------- Load weights ----------------- #
    # If don't have weights, comment later code
    ckpt_dir = "/data/weights/Qwen3-0.6B"
    model = Qwen3ForCausalLM.from_pretrained(ckpt_dir)
    past = None
    with torch.no_grad():
        # 1) prefill: run full prompt ONCE, cache everything
        out = model(x, past_key_values=None, use_cache=True)
        past = out.past_key_values

        # next token from prompt end
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        print("first next_token:", next_token.item())

        # 2) decode: decode one token at a time using cache
        for step in range(5):
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values

            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            print(f"step={step} next_token={next_token.item()}")
