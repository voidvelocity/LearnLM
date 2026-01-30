import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3 import Qwen3ForCausalLM

ckpt_dir = "/data/weights/Qwen3-0.6B"


def compare_logits(my_model, hf_model):
    # token ids must be torch.long
    x = torch.tensor([[33464, 6832, 374]], dtype=torch.long)

    with torch.no_grad():
        my_out = my_model(x, use_cache=False)
        my_logits = my_out.logits

    with torch.no_grad():
        hf_out = hf_model(x, use_cache=False)
        hf_logits = hf_out.logits

    print("my logits(min, mean, max):", my_logits.min().item(), my_logits.mean().item(), my_logits.max().item())
    print("hf logits(min, mean, max):", hf_logits.min().item(), hf_logits.mean().item(), hf_logits.max().item())
    print("Max difference:", (hf_logits - my_logits).abs().max().item())


def generate_token_my_model(model, prompt: str, max_new_tokens=5, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device=device, dtype=torch.long)
    generated = input_ids.clone()

    print("input_ids:", input_ids)

    past = None

    with torch.no_grad():
        # ---- PREFILL (full prompt once)
        out = model(input_ids, past_key_values=None, use_cache=True)
        past = out.past_key_values

        # first next token from prompt end
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)
        print("prefill next_token:", next_token.item())
        print("text:", tokenizer.decode(generated[0], skip_special_tokens=True))

        # ---- DECODE (1 token each step)
        for step in range(max_new_tokens - 1):
            print(f"--- decode step [{step}] ---")

            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values

            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            print("token id:", generated)
            print("text:", tokenizer.decode(generated[0], skip_special_tokens=True))

            if next_token.item() == tokenizer.eos_token_id:
                break


def generate_token_hf_model(model, prompt: str, max_new_tokens=5, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device=device, dtype=torch.long)
    generated = input_ids.clone()

    print("input_ids:", input_ids)

    past = None

    with torch.no_grad():
        # PREFILL
        out = model(input_ids, past_key_values=None, use_cache=True)
        past = out.past_key_values

        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        print("prefill next_token:", next_token.item())
        print("text:", tokenizer.decode(generated[0], skip_special_tokens=True))

        # DECODE
        for step in range(max_new_tokens - 1):
            print(f"--- decode step [{step}] ---")

            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values

            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            print("token id:", generated)
            print("text:", tokenizer.decode(generated[0], skip_special_tokens=True))

            if next_token.item() == tokenizer.eos_token_id:
                break


if __name__ == "__main__":
    device = "cpu"

    my_model = Qwen3ForCausalLM.from_pretrained(
        ckpt_dir,
        device=device
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    prompt = "Deep learning is"

    print(f"\n============ my qwen3 model ============\n")
    generate_token_my_model(my_model, prompt, max_new_tokens=5, device=device)

    print(f"\n============ hf qwen3 model ============\n")
    generate_token_hf_model(hf_model, prompt, max_new_tokens=5, device=device)

    print(f"\n============ compare logits ============\n")
    compare_logits(my_model, hf_model)
