from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128

    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0

    max_position_embeddings: int = 40960
    rope_theta: int = 1_000_000
    rope_scaling: Optional[dict] = None

    sliding_window: Optional[int] = None
    max_window_layers: int = 28
    use_sliding_window: bool = False

    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    bos_token_id: int = 151643
    eos_token_id: int = 151645

    tie_word_embeddings: bool = True
    use_cache: bool = True
    torch_dtype: str = "bfloat16"

    architectures: List[str] = field(
        default_factory=lambda: ["Qwen3ForCausalLM"]
    )

    # forward-compat bucket
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        obj = cls(**kwargs)
        obj.extra.update(extra)
        return obj


@dataclass
class CausalLMOutput:
    logits: "torch.Tensor"
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
