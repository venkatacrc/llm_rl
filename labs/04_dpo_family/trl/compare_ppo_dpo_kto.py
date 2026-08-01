"""Lab 04 (at scale): compare PPO (lab 03) vs DPO vs KTO on the same reward-vs-KL axes.

Loads whichever of these checkpoints exist:
    - checkpoints/sft_qwen2.5-0.5b_tldr        (lab 01, also the reference policy)
    - checkpoints/ppo_qwen2.5-0.5b_tldr        (lab 03)
    - checkpoints/dpo_qwen2.5-0.5b_tldr        (this lab)
    - checkpoints/kto_qwen2.5-0.5b_tldr        (this lab)

...generates completions for a shared set of held-out TL;DR prompts, scores them with
the lab-02 reward model, and estimates each model's mean per-token KL divergence from
the SFT reference. Produces a scatter plot (reward vs KL) -- the "reward-vs-KL
frontier" every alignment method is implicitly trading off against, revisited more
formally in lab 08.

Run (after training whichever of the above you want to compare):
    python labs/04_dpo_family/trl/compare_ppo_dpo_kto.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.paths import REWARD_MODEL_NAME, SFT_MODEL_NAME, checkpoint_dir, run_dir  # noqa: E402
from common.utils import banner, decode_batch, left_pad_batch, set_seed  # noqa: E402

N_EVAL_PROMPTS = 128
MAX_NEW_TOKENS = 64
CANDIDATES = {
    "SFT (reference)": SFT_MODEL_NAME,
    "PPO": "ppo_qwen2.5-0.5b_tldr",
    "DPO": "dpo_qwen2.5-0.5b_tldr",
    "KTO": "kto_qwen2.5-0.5b_tldr",
}


@torch.no_grad()
def generate_and_score(model, ref_model, reward_model, reward_tokenizer, tokenizer, prompts, device):
    from common.rewards import reward_model_score

    input_ids, attn_mask = left_pad_batch(tokenizer, prompts, device, max_length=512)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    full_mask = torch.cat([attn_mask, (out[:, input_ids.shape[1] :] != tokenizer.pad_token_id).long()], dim=1)
    completions = decode_batch(tokenizer, out[:, input_ids.shape[1] :], full_mask[:, input_ids.shape[1] :])
    rewards = reward_model_score(reward_model, reward_tokenizer, prompts, completions, device=device)

    # Approximate mean per-token KL(model || ref) over the generated tokens.
    policy_logits = model(input_ids=out, attention_mask=full_mask).logits[:, input_ids.shape[1] - 1 : -1, :]
    ref_logits = ref_model(input_ids=out, attention_mask=full_mask).logits[:, input_ids.shape[1] - 1 : -1, :]
    gen_tokens = out[:, input_ids.shape[1] :]
    gen_mask = full_mask[:, input_ids.shape[1] :].float()
    policy_logp = torch.gather(F.log_softmax(policy_logits, -1), 2, gen_tokens.unsqueeze(-1)).squeeze(-1)
    ref_logp = torch.gather(F.log_softmax(ref_logits, -1), 2, gen_tokens.unsqueeze(-1)).squeeze(-1)
    per_token_kl = (policy_logp - ref_logp) * gen_mask
    mean_kl = (per_token_kl.sum(-1) / gen_mask.sum(-1).clamp(min=1)).mean().item()

    return sum(rewards) / len(rewards), mean_kl, completions[:3]


def main() -> None:
    set_seed(0)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner("Lab 04: reward-vs-KL comparison across PPO / DPO / KTO")

    reward_dir = checkpoint_dir(REWARD_MODEL_NAME, create=False)
    if not (reward_dir.exists() and any(reward_dir.iterdir())):
        raise SystemExit(f"No reward model at {reward_dir}. Run labs/02_reward_modeling/trl/train_rm_trl.py first.")
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_dir, num_labels=1).to(device).eval()
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_dir)

    sft_dir = checkpoint_dir(SFT_MODEL_NAME, create=False)
    if not (sft_dir.exists() and any(sft_dir.iterdir())):
        raise SystemExit(f"No SFT model at {sft_dir}. Run labs/01_sft_warmup/trl/train_sft_trl.py first.")
    tokenizer = AutoTokenizer.from_pretrained(sft_dir, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_model = AutoModelForCausalLM.from_pretrained(sft_dir, dtype=torch.bfloat16).to(device).eval()

    prompts = load_dataset("trl-lib/tldr", split="validation").select(range(N_EVAL_PROMPTS))["prompt"]

    results = {}
    for label, ckpt_name in CANDIDATES.items():
        ckpt_dir = checkpoint_dir(ckpt_name, create=False)
        if not (ckpt_dir.exists() and any(ckpt_dir.iterdir())):
            print(f"[skip] {label}: no checkpoint at {ckpt_dir}")
            continue
        print(f"\n[{label}] generating + scoring from {ckpt_dir} ...")
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16).to(device).eval()
        mean_reward, mean_kl, samples = generate_and_score(
            model, ref_model, reward_model, reward_tokenizer, tokenizer, prompts, device
        )
        results[label] = (mean_reward, mean_kl)
        print(f"  mean_reward={mean_reward:.3f}  mean_kl_vs_sft={mean_kl:.4f}")
        print(f"  sample: {samples[0]!r}")
        del model
        torch.cuda.empty_cache()

    if len(results) < 2:
        print("\nNeed at least 2 checkpoints to plot a comparison -- train more of PPO/DPO/KTO and re-run.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, (reward, kl) in results.items():
        ax.scatter(kl, reward, s=80)
        ax.annotate(label, (kl, reward), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("mean KL(policy || SFT reference) on generated tokens")
    ax.set_ylabel("mean reward model score")
    ax.set_title("Reward vs. KL frontier: PPO vs DPO vs KTO")
    ax.grid(alpha=0.3)
    out_path = run_dir("lab04_dpo_family") / "reward_vs_kl.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved comparison plot to {out_path}")
    print(
        "\nA method that's strictly better sits up-and-to-the-left (higher reward for the same "
        "or less KL drift). If PPO/DPO/KTO land on roughly the same frontier, that's the "
        "well-known empirical finding that these methods are more similar in what they achieve "
        "than their derivations suggest -- see lab 08 for why chasing reward without bound is a "
        "trap regardless of which algorithm gets you there."
    )


if __name__ == "__main__":
    main()
