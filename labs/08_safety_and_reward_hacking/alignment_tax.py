"""Lab 08, part 3: measuring the "alignment tax".

RLHF/RLVR optimizes a model towards a specific target (be preferred by a reward model,
get math answers right, refuse harmful requests...) which can trade off against general
capability -- a base pretrained model is a jack-of-all-trades next-token predictor;
narrowing its distribution towards "what humans rate highly" or "what one reward model
likes" can, and empirically sometimes does, cost a bit of raw capability elsewhere.
That cost is informally called the "alignment tax".

This script scores every available checkpoint from labs 01-06 on a small held-out MMLU
subset using *log-likelihood* multiple-choice scoring (no generation -- for each
question we just compare the model's log-probability of "A"/"B"/"C"/"D" as the next
token, and take the argmax), which is fast and doesn't depend on instruction-following
quality the way generation-based eval would.

Run (after training whichever of labs 01/03/04 you want to compare):
    python labs/08_safety_and_reward_hacking/alignment_tax.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.paths import SFT_MODEL_NAME, checkpoint_dir, run_dir  # noqa: E402
from common.utils import banner, set_seed  # noqa: E402

N_EVAL_QUESTIONS = 200
LETTERS = ["A", "B", "C", "D"]

CANDIDATES = {
    "Base (Qwen2.5-0.5B, no alignment)": None,  # loaded directly from the hub, see below
    "SFT": SFT_MODEL_NAME,
    "PPO": "ppo_qwen2.5-0.5b_tldr",
    "DPO": "dpo_qwen2.5-0.5b_tldr",
    "KTO": "kto_qwen2.5-0.5b_tldr",
}
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"


def format_question(example) -> str:
    choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, example["choices"]))
    return f"Question: {example['question']}\n{choices}\nAnswer:"


@torch.no_grad()
def score_mmlu(model, tokenizer, examples, device) -> float:
    letter_ids = [tokenizer(f" {letter}", add_special_tokens=False)["input_ids"][0] for letter in LETTERS]
    correct = 0
    for ex in examples:
        prompt = format_question(ex)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=input_ids).logits[0, -1, :]  # next-token distribution after "Answer:"
        letter_logits = logits[letter_ids]
        pred = int(torch.argmax(letter_logits).item())
        correct += int(pred == ex["answer"])
    return correct / len(examples)


def main() -> None:
    set_seed(0)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner("Lab 08: alignment tax (MMLU log-likelihood accuracy across checkpoints)")

    mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(N_EVAL_QUESTIONS))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    results = {}
    for label, ckpt_name in CANDIDATES.items():
        if ckpt_name is None:
            model_path = BASE_MODEL_ID
        else:
            ckpt_dir = checkpoint_dir(ckpt_name, create=False)
            if not (ckpt_dir.exists() and any(ckpt_dir.iterdir())):
                print(f"[skip] {label}: no checkpoint at {ckpt_dir}")
                continue
            model_path = str(ckpt_dir)

        print(f"\n[{label}] scoring from {model_path} ...")
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()
        acc = score_mmlu(model, tokenizer, mmlu, device)
        results[label] = acc
        print(f"  MMLU (n={N_EVAL_QUESTIONS}) accuracy: {acc:.3f}")
        del model
        torch.cuda.empty_cache()

    if not results:
        print("\nNo checkpoints found -- train at least the lab-01 SFT model first.")
        return

    print(f"\n{'model':>35s} | {'MMLU acc':>9s} | {'delta vs base':>14s}")
    base_acc = results.get("Base (Qwen2.5-0.5B, no alignment)")
    for label, acc in results.items():
        delta = f"{acc - base_acc:+.3f}" if base_acc is not None else "n/a"
        print(f"{label:>35s} | {acc:>9.3f} | {delta:>14s}")

    out_path = run_dir("lab08_alignment_tax") / "mmlu_results.csv"
    with open(out_path, "w") as f:
        f.write("model,mmlu_accuracy\n")
        for label, acc in results.items():
            f.write(f"{label},{acc:.4f}\n")
    print(f"\nSaved results to {out_path}")
    print(
        "\nA 0.5B model is small enough that MMLU accuracy will be noisy/close to chance (25%) for "
        "all variants -- the point of this script is the *methodology* (log-likelihood scoring, "
        "before/after comparison) more than a definitive tax measurement at this scale. Re-run "
        "against lab 07's 7-8B capstone checkpoint for a more meaningful signal."
    )


if __name__ == "__main__":
    main()
