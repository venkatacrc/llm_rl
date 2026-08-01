"""Lab 05 (scratch): GRPO for RLVR (verifiable rewards), written by hand.

GRPO (Group Relative Policy Optimization, used to train DeepSeek-R1) removes the value
function/critic entirely -- the single biggest structural difference from PPO (lab 03).
Instead of learning $V_\\psi(s)$ to build a baseline, GRPO samples a *group* of $G$
completions for the same prompt and uses the group's own mean/std as the baseline:

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)          for i = 1..G

This is cheap (no extra network, no GAE), works well when rewards are *verifiable*
(exactly right/wrong, so within a group you'll usually get a mix of successes and
failures once the policy is decent), and is the standard recipe for RLVR
(RL from Verifiable Rewards): the reward isn't a learned model at all, it's a rule
("did the model's final answer match the ground truth?").

Everything else mirrors PPO's clipped-surrogate update, but the KL term is folded
directly into the *loss* (not into a per-token reward) using Schulman's unbiased k3
estimator, which is what the GRPO paper (and TRL's `GRPOTrainer`) actually use:

    KL_k3(policy || ref) ~= exp(logp_ref - logp_policy) - (logp_ref - logp_policy) - 1

Task: single-digit addition ("3 + 5 = "), reward = 1.0 if the model's completion
contains the correct sum, else 0.0 -- a minimal but genuine verifiable-reward task:
GPT-2 has some baseline ability at this (arithmetic facts show up a lot in web text),
so groups will contain a *mix* of right/wrong answers early in training, which is
exactly what GRPO needs to get a useful advantage signal. If every sample in a group
gets the same reward (all right, or -- more likely early on -- all wrong), that group's
advantages are all ~0 and contributes no gradient: this is a real, important property
of RLVR to see happen, not a bug (see "things to try" below).

Run:
    python labs/05_grpo_rlvr_reasoning/scratch/train_grpo_scratch.py
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
N_PROMPTS_PER_STEP = 16  # -> batch size = N_PROMPTS_PER_STEP * GROUP_SIZE
N_STEPS = 300
RESPONSE_TOKENS = 8
GRPO_EPOCHS = 2
CLIP_EPS = 0.2
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


def left_pad(id_lists: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
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
    logprobs = torch.gather(log_probs_all, 2, target.unsqueeze(-1)).squeeze(-1)
    return logprobs


def k3_kl(policy_logprob: torch.Tensor, ref_logprob: torch.Tensor) -> torch.Tensor:
    """Schulman's unbiased, always-non-negative KL(policy || ref) estimator."""
    log_ratio = ref_logprob - policy_logprob
    return torch.exp(log_ratio) - log_ratio - 1


def main() -> None:
    set_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner(f"Lab 05 (scratch): GRPO on {MODEL_NAME} / single-digit addition (RLVR)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    ref_policy = copy.deepcopy(policy).to(device).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(policy.parameters(), lr=LR)
    logger = RunLogger(
        run_dir("lab05_grpo_scratch"), "grpo_scratch", config={"group_size": GROUP_SIZE, "kl_beta": KL_BETA}
    )

    rng = random.Random(0)

    for step in range(N_STEPS):
        # ---------------- 1. Sample a batch of problems, each replicated G times ----------------
        problems = [make_problem(rng) for _ in range(N_PROMPTS_PER_STEP)]
        prompt_texts, golds = zip(*problems)
        prompt_ids_single = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompt_texts]
        # replicate each prompt G times so groups sit contiguously in the batch
        prompt_ids_batch = [ids for ids in prompt_ids_single for _ in range(GROUP_SIZE)]
        golds_batch = [g for g in golds for _ in range(GROUP_SIZE)]

        prompt_ids, prompt_mask = left_pad(prompt_ids_batch, tokenizer.pad_token_id, device)
        prompt_len = prompt_ids.shape[1]

        # ---------------- 2. Rollout: G samples per prompt ----------------
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
        response_ids = gen_out[:, prompt_len:]
        full_mask = torch.cat([prompt_mask, torch.ones_like(response_ids)], dim=1)
        response_texts = decode_batch(tokenizer, response_ids)

        # ---------------- 3. Verifiable reward + group-normalized advantage ----------------
        rewards = torch.tensor(
            [1.0 if extract_answer(t) == g else 0.0 for t, g in zip(response_texts, golds_batch)], device=device
        )
        rewards_grouped = rewards.view(N_PROMPTS_PER_STEP, GROUP_SIZE)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True)
        advantages_grouped = (rewards_grouped - group_mean) / (group_std + 1e-4)
        advantages = advantages_grouped.view(-1)  # (B,), one scalar per *sequence*, broadcast over its tokens

        # ---------------- 4. Old policy + reference log-probs (no grad) ----------------
        with torch.no_grad():
            old_logprobs = get_response_logprobs(policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)
            ref_logprobs = get_response_logprobs(ref_policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)

        # ---------------- 5. GRPO update (clipped surrogate + loss-level KL term) ----------------
        policy.train()
        for _ in range(GRPO_EPOCHS):
            new_logprobs = get_response_logprobs(policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)
            ratio = torch.exp(new_logprobs - old_logprobs)
            adv = advantages.unsqueeze(1)  # broadcast the per-sequence advantage over response tokens

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            policy_term = torch.min(surr1, surr2)

            kl_term = k3_kl(new_logprobs, ref_logprobs)
            per_token_loss = -(policy_term - KL_BETA * kl_term)
            loss = per_token_loss.mean()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

        if step % LOG_EVERY == 0:
            zero_adv_groups = (group_std.squeeze(-1) < 1e-6).float().mean().item()
            logger.log(
                step,
                mean_reward=rewards.mean().item(),
                loss=loss.item(),
                mean_kl=kl_term.mean().item(),
                zero_signal_group_frac=zero_adv_groups,
            )
            print(f"    e.g. '{prompt_texts[0]}' -> '{response_texts[0].strip()}' (gold={golds[0]})")

    logger.plot(
        ["mean_reward", "mean_kl", "zero_signal_group_frac"], title="GRPO on GPT-2 / single-digit addition"
    )

    out_dir = run_dir("lab05_grpo_scratch") / "final_model"
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved GRPO-tuned policy to {out_dir}")
    print(
        "\nmean_reward (exact-match accuracy on single-digit addition) should climb from GPT-2's "
        "weak baseline towards a much higher accuracy. zero_signal_group_frac shows what fraction "
        "of groups had zero variance in reward (all-right or all-wrong) and therefore contributed "
        "no learning signal that step -- watch it shrink as the model gets better (more groups "
        "become 'mostly right, occasionally wrong', which is where GRPO's signal is richest)."
    )


if __name__ == "__main__":
    main()
