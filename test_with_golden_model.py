import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch.nn.functional as F

from qwen3 import Qwen3ForCausalLM


ckpt_dir = "/data/weights/Qwen3-0.6B"

def compare_logits(my_model, hf_model):
    x = torch.tensor([[33464,  6832,   374]], dtype=torch.int32)

    with torch.no_grad():
        my_logits = my_model(x).logits

    with torch.no_grad():
        hf_logits = hf_model(x).logits

    print("my logits(min, mean, max):", my_logits.min(), my_logits.mean(), my_logits.max())
    print("hf logits(min, mean, max):", hf_logits.min(), hf_logits.mean(), hf_logits.max())
    print("Max difference:", (hf_logits - my_logits).abs().max())


def generate_token(model, prompt: str, max_new_tokens=5):
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print("input id:", input_ids)
    generated = input_ids.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward
            print(f"--- step [{step}] ---")
            logits = model(generated).logits         # [B, T, vocab_size]
            next_logits = logits[:, -1, :]           # [B, vocab_size]
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [1, 1]

            # Append
            generated = torch.cat([generated, next_token], dim=1)
            print("token id:", generated)
            output_text = tokenizer.decode(
                generated[0],
                skip_special_tokens=True
            )
            print("text:", output_text)

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break


if __name__ == "__main__":

    my_model = Qwen3ForCausalLM.from_pretrained(
        ckpt_dir,
        device="cpu"
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    prompt = "Deep learning is"

    print(f"\n============ my qwen3 model ============\n")
    generate_token(my_model, prompt)

    print(f"\n============ hf qwen3 model ============\n")
    generate_token(my_model, prompt)

    print(f"\n============ compare logits ============\n")
    compare_logits(my_model, hf_model)

