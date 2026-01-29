import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # q = q.view(B, H, 2, T, D).reshape(B, H, 2 * T, D)  # [B, 2H, T, D] -> [B, H, 2, T, D]

        # q: [B, 2H, T, D]
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
        # [B, 2H, T, T] @ [B, H, T, D] -> [B, 2H, T, D]
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

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)


def load_weight(model):
    pass

if __name__ == "__main__":
    my_model = Qwen3ForCausalLM()
    # from transformers import AutoModelForCausalLM
    # hf_model = AutoModelForCausalLM.from_pretrained(
    #     "/data/weights/Qwen3-0.6B",
    #     torch_dtype=torch.float32,
    #     device_map="cpu",
    # )
    # my_model = hf_model

    my_model.eval()

    from safetensors.torch import load_file
    ckpt_path = "/data/weights/Qwen3-0.6B/model.safetensors"
    state_dict = load_file(ckpt_path, device="cpu")
    print(list(state_dict.keys())[:20])

    missing, unexpected = my_model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)


    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/data/weights/Qwen3-0.6B",
        trust_remote_code=True,
    )
    prompt = "Introduce deep learning."
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    max_new_tokens = 10
    temperature = 1.0
    top_k = None         #50

    generated = input_ids.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward
            logits = my_model(generated)      # [1, T, vocab]  # 如果用 hf_model，改成 logits = my_model(generated).logits
            next_logits = logits[:, -1, :]    # [1, vocab]

            # Temperature
            next_logits /= temperature

            # Top-k filtering (optional but recommended)
            if top_k is not None:
                values, indices = torch.topk(next_logits, top_k)
                probs = torch.zeros_like(next_logits).scatter_(1, indices, values)
                probs = F.softmax(probs, dim=-1)
            else:
                probs = F.softmax(next_logits, dim=-1)

            # Sample
            # next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [1, 1]

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            output_text = tokenizer.decode(
                generated[0],
                skip_special_tokens=True
            )
            print("---", output_text)

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    # from transformers import AutoModelForCausalLM
    # hf_model = AutoModelForCausalLM.from_pretrained(
    #     "/data/weights/Qwen3-0.6B",
    #     torch_dtype=torch.float32,
    #     device_map="cpu",
    # )
    # hf_model.eval()


    # torch.manual_seed(0)
    # x = torch.randint(0, 151936, (1, 8))

    # with torch.no_grad():
    #     hf_logits = hf_model(x).logits
    # print(hf_logits.shape)

    # with torch.no_grad():
    #     # x.shape: [1, 5]
    #     my_logits = my_model(x)  # [B, T, vocab_size] [1, 5, 151936]
    # print(my_logits.shape)
    # print("max abs diff:", (hf_logits - my_logits).abs().max())
    # print("value:", hf_logits.max(), hf_logits.min(), hf_logits.mean())



"""
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm()
          (k_norm): Qwen3RMSNorm()
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
        )
        (input_layernorm): Qwen3RMSNorm()
        (post_attention_layernorm): Qwen3RMSNorm()
      )
    )
    (norm): Qwen3RMSNorm()
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)
"""
