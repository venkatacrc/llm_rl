"""Lab 08, part 2: the reward-vs-KL frontier.

Every RLHF algorithm in this curriculum is, explicitly or implicitly, solving
`max_theta E[reward] - beta * KL(policy || reference)` for some beta (PPO's KL penalty,
GRPO's loss-level KL term, DPO/KTO's implicit beta). This script makes that trade-off
literal: train several short runs at different `beta` values (same task/reward as lab
03 -- sentiment-classifier reward on GPT-2-imdb), and plot the achieved
(reward, KL-from-reference) pair for each. The result is a Pareto frontier: for a fixed
amount of "how far from the original model are we willing to drift" (x-axis), there's a
best achievable reward (y-axis) -- and pushing reward higher than that frontier at a
given beta usually means you've found a hack, not a genuine improvement (see
`reward_hacking_demo.py`).

Run (a few minutes: several short training runs):
    python labs/08_safety_and_reward_hacking/kl_reward_frontier.py
"""

from __future__ import annotations

import copy
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.paths import run_dir  # noqa: E402
from common.rewards import SentimentReward  # noqa: E402
from common.utils import banner, decode_batch, set_seed  # noqa: E402
from reward_hacking_demo import build_prompt_ids, left_pad  # noqa: E402

POLICY_MODEL_NAME = "lvwerra/gpt2-imdb"
BETAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
BATCH_SIZE = 64
RESPONSE_TOKENS = 32
N_STEPS = 60
LR = 1.41e-5


def get_response_logprobs(model, full_ids, full_mask, prompt_len, response_len):
    logits = model(input_ids=full_ids, attention_mask=full_mask).logits
    shifted = logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
    log_probs_all = F.log_softmax(shifted, dim=-1)
    target = full_ids[:, prompt_len : prompt_len + response_len]
    return torch.gather(log_probs_all, 2, target.unsqueeze(-1)).squeeze(-1)


def train_one_beta(beta: float, tokenizer, reward_model, all_prompts, device):
    from transformers import AutoModelForCausalLM

    policy = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(device)
    ref_policy = copy.deepcopy(policy).to(device).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    optim = torch.optim.AdamW(policy.parameters(), lr=LR)

    last_reward, last_kl = None, None
    for step in range(N_STEPS):
        batch_prompts = all_prompts[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        prompt_ids, prompt_mask = left_pad(batch_prompts, tokenizer.pad_token_id, device)
        prompt_len = prompt_ids.shape[1]

        policy.eval()
        with torch.no_grad():
            gen_out = policy.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=RESPONSE_TOKENS,
                min_new_tokens=RESPONSE_TOKENS,
                do_sample=True,
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        response_ids = gen_out[:, prompt_len:]
        full_mask = torch.cat([prompt_mask, torch.ones_like(response_ids)], dim=1)
        full_texts = decode_batch(tokenizer, gen_out, full_mask)

        rewards = torch.tensor(reward_model(completions=full_texts), device=device)

        with torch.no_grad():
            old_logprobs = get_response_logprobs(policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)
            ref_logprobs = get_response_logprobs(ref_policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)
        per_token_kl = old_logprobs - ref_logprobs

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        policy.train()
        new_logprobs = get_response_logprobs(policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)
        seq_logprob = new_logprobs.sum(dim=1)
        loss = -(advantages.detach() * seq_logprob).mean() + beta * per_token_kl.mean()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()

        last_reward, last_kl = rewards.mean().item(), per_token_kl.mean().item()
        if step % 20 == 0:
            print(f"    [beta={beta}] step {step:>3d}  reward={last_reward:.3f}  kl={last_kl:.4f}")

    del policy, ref_policy
    torch.cuda.empty_cache()
    return last_reward, last_kl


def main() -> None:
    set_seed(0)
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner("Lab 08: reward-vs-KL frontier sweep")

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    reward_model = SentimentReward(device=device)
    all_prompts = build_prompt_ids(tokenizer, n=BATCH_SIZE * N_STEPS, seed=1)

    results = []
    for beta in BETAS:
        print(f"\n--- Training with beta={beta} ---")
        reward, kl = train_one_beta(beta, tokenizer, reward_model, all_prompts, device)
        results.append((beta, reward, kl))

    print(f"\n{'beta':>8s} | {'final reward':>13s} | {'final KL':>10s}")
    for beta, reward, kl in results:
        print(f"{beta:>8.3f} | {reward:>13.3f} | {kl:>10.4f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    kls = [r[2] for r in results]
    rewards = [r[1] for r in results]
    ax.plot(kls, rewards, marker="o")
    for beta, reward, kl in results:
        ax.annotate(f"beta={beta}", (kl, reward), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("mean per-token KL(policy || reference)")
    ax.set_ylabel("mean reward (sentiment classifier)")
    ax.set_title("Reward-vs-KL frontier across KL coefficients")
    ax.grid(alpha=0.3)
    out_path = run_dir("lab08_kl_reward_frontier") / "frontier.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    print(
        "\nSmaller beta -> more KL drift allowed -> (usually) higher achievable reward, up to a "
        "point of diminishing/negative returns as beta -> 0 and the policy starts exploiting the "
        "reward model off-distribution (compare against reward_hacking_demo.py's beta=0 run)."
    )


if __name__ == "__main__":
    main()
