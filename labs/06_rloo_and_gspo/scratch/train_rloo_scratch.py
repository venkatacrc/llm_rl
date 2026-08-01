"""Lab 06 (scratch): RLOO -- REINFORCE with a Leave-One-Out baseline.

RLOO strips GRPO (lab 05) down further: no PPO-style clipping/importance ratio at all,
just the original REINFORCE policy gradient,

    grad J(theta) = E[ (R - b) * grad logpi_theta(y|x) ]

with a baseline `b` chosen to reduce variance without introducing bias. The
Leave-One-Out baseline, given a group of G samples for the same prompt, is:

    b_i = mean(r_1, ..., r_G excluding r_i) = (sum_j r_j - r_i) / (G - 1)
    A_i = r_i - b_i

This is *exactly* unbiased (b_i doesn't depend on r_i, so subtracting it doesn't change
the expected gradient), unlike GRPO's group-*mean* baseline (which includes r_i in its
own baseline -- a small O(1/G) bias in exchange for slightly simpler bookkeeping and, by
GRPO's own convention, dividing by the group std too, which is a genuinely different,
scale-normalizing choice). This script computes and logs all three candidate baselines
side-by-side on the *same* rollouts each step, purely to make the bias/variance
trade-off concrete, then trains using the (unbiased) LOO advantage.

Task: the same single-digit-addition RLVR task as lab 05's GRPO script, so you can
directly compare training curves between the two labs.

Run:
    python labs/06_rloo_and_gspo/scratch/train_rloo_scratch.py
"""

from __future__ import annotations

import copy
import os
import random
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.paths import run_dir  # noqa: E402
from common.utils import RunLogger, banner, decode_batch, set_seed  # noqa: E402

MODEL_NAME = "gpt2"
GROUP_SIZE = 8
N_PROMPTS_PER_STEP = 16
N_STEPS = 300
RESPONSE_TOKENS = 8
KL_BETA = 0.02
LR = 1e-5
LOG_EVERY = 5

_NUM_RE = re.compile(r"-?\d+")


def make_problem(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(0, 9), rng.randint(0, 9)
    return f"{a} + {b} = ", a + b


def extract_answer(text: str) -> int | None:
    m = _NUM_RE.search(text)
    return int(m.group()) if m else None


def left_pad(id_lists, pad_id, device):
    max_len = max(len(ids) for ids in id_lists)
    input_ids, attn_mask = [], []
    for ids in id_lists:
        pad_n = max_len - len(ids)
        input_ids.append([pad_id] * pad_n + ids)
        attn_mask.append([0] * pad_n + [1] * len(ids))
    return torch.tensor(input_ids, device=device), torch.tensor(attn_mask, device=device)


def get_response_logprobs(model, full_ids, full_mask, prompt_len, response_len):
    logits = model(input_ids=full_ids, attention_mask=full_mask).logits
    shifted = logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
    log_probs_all = F.log_softmax(shifted, dim=-1)
    target = full_ids[:, prompt_len : prompt_len + response_len]
    return torch.gather(log_probs_all, 2, target.unsqueeze(-1)).squeeze(-1)


def k3_kl(policy_logprob, ref_logprob):
    log_ratio = ref_logprob - policy_logprob
    return torch.exp(log_ratio) - log_ratio - 1


def leave_one_out_advantage(rewards_grouped: torch.Tensor) -> torch.Tensor:
    G = rewards_grouped.shape[1]
    group_sum = rewards_grouped.sum(dim=1, keepdim=True)
    baseline = (group_sum - rewards_grouped) / (G - 1)
    return rewards_grouped - baseline


def main() -> None:
    set_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner(f"Lab 06 (scratch): RLOO on {MODEL_NAME} / single-digit addition (RLVR)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    ref_policy = copy.deepcopy(policy).to(device).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=LR)
    logger = RunLogger(run_dir("lab06_rloo_scratch"), "rloo_scratch", config={"group_size": GROUP_SIZE})

    rng = random.Random(0)

    for step in range(N_STEPS):
        problems = [make_problem(rng) for _ in range(N_PROMPTS_PER_STEP)]
        prompt_texts, golds = zip(*problems)
        prompt_ids_single = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompt_texts]
        prompt_ids_batch = [ids for ids in prompt_ids_single for _ in range(GROUP_SIZE)]
        golds_batch = [g for g in golds for _ in range(GROUP_SIZE)]

        prompt_ids, prompt_mask = left_pad(prompt_ids_batch, tokenizer.pad_token_id, device)
        prompt_len = prompt_ids.shape[1]

        policy.eval()
        with torch.no_grad():
            gen_out = policy.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=RESPONSE_TOKENS,
                min_new_tokens=RESPONSE_TOKENS,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id,
            )
        response_texts = decode_batch(tokenizer, gen_out[:, prompt_len:])
        full_mask = torch.cat([prompt_mask, torch.ones_like(gen_out[:, prompt_len:])], dim=1)

        rewards = torch.tensor(
            [1.0 if extract_answer(t) == g else 0.0 for t, g in zip(response_texts, golds_batch)], device=device
        )
        rewards_grouped = rewards.view(N_PROMPTS_PER_STEP, GROUP_SIZE)

        # --- Three baseline choices, computed side-by-side for comparison ---
        adv_naive = rewards_grouped - 0.0  # no baseline at all (vanilla REINFORCE)
        adv_group_mean = rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)  # GRPO-style (biased, O(1/G))
        adv_loo = leave_one_out_advantage(rewards_grouped)  # RLOO (unbiased)

        advantages = adv_loo.view(-1)  # this is what we actually train with

        with torch.no_grad():
            ref_logprobs = get_response_logprobs(ref_policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)

        # --- Single vanilla policy-gradient step (no clipping, no reused-data epochs -- ---
        # --- this IS the "RL" in REINFORCE: one on-policy gradient step per rollout).  ---
        policy.train()
        new_logprobs = get_response_logprobs(policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)
        seq_logprob = new_logprobs.sum(dim=1)  # log pi(y|x) = sum of per-token log-probs
        pg_loss = -(advantages.detach() * seq_logprob).mean()

        kl_term = k3_kl(new_logprobs, ref_logprobs)
        loss = pg_loss + KL_BETA * kl_term.mean()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()

        if step % LOG_EVERY == 0:
            logger.log(
                step,
                mean_reward=rewards.mean().item(),
                loss=loss.item(),
                mean_kl=kl_term.mean().item(),
                var_naive=adv_naive.var().item(),
                var_group_mean=adv_group_mean.var().item(),
                var_loo=adv_loo.var().item(),
            )
            print(f"    e.g. '{prompt_texts[0]}' -> '{response_texts[0].strip()}' (gold={golds[0]})")

    logger.plot(["mean_reward", "var_naive", "var_group_mean", "var_loo"], title="RLOO on GPT-2 / single-digit addition")

    out_dir = run_dir("lab06_rloo_scratch") / "final_model"
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved RLOO-tuned policy to {out_dir}")
    print(
        "\nvar_naive should be visibly higher than var_group_mean and var_loo throughout training -- "
        "that's the concrete effect of *any* reasonable baseline. var_group_mean and var_loo are "
        "usually close (both subtract ~the group mean); the difference between them is bias, not "
        "variance -- see lab06/README.md for the exact argument."
    )


if __name__ == "__main__":
    main()
