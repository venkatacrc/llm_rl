"""Lab 00, part 4: load an instruct model and generate -- the simplest possible
end-to-end check that transformers + chat templates + your GPU all work together
before you start training anything.

    python labs/00_setup_and_gpu_tour/hello_generate.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.utils import banner, set_seed  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "In one sentence, explain what reinforcement learning from human feedback is.",
    "Write a haiku about GPUs.",
    "What is 17 * 24? Think step by step.",
]


def main() -> None:
    set_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    banner(f"Loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa").to(
        "cuda"
    )
    model.eval()
    print(f"Loaded on {next(model.parameters()).device}, dtype={next(model.parameters()).dtype}")

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            "cuda"
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        with torch.no_grad():
            out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9)
        response = tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
        print(f"\n> {prompt}\n{response.strip()}\n{'-' * 60}")

    print("\nIf you saw three coherent responses above, the stack works end-to-end.")


if __name__ == "__main__":
    main()
