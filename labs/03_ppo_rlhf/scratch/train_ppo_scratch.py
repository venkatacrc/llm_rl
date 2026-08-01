"""Lab 03 (scratch): PPO for RLHF, written by hand.

The classic InstructGPT-style toy task: start from a GPT-2 already fine-tuned on IMDB
movie reviews (`lvwerra/gpt2-imdb`, so it's already "SFT'd" for this task -- see lab 01
for how that fine-tuning step itself works), give it short review prefixes, and use PPO
to shift it towards generating *positive*-sentiment continuations, as judged by a small
pretrained sentiment classifier (`lvwerra/distilbert-imdb`) acting as the reward model
(see lab 02 for how a reward model like this is trained).

This script implements every piece of PPO for language models by hand:

  1. Rollout: sample responses from the current policy for a batch of prompts.
  2. Reward assignment: R(x,y) = reward_model(x,y) at the final token, plus a per-token
     `-beta * KL(policy || reference)` penalty at every token (this is what keeps the
     policy from drifting arbitrarily far from the reference/SFT model just to farm
     reward -- see lab 08 for what happens when you turn beta down too far).
  3. Advantage estimation: Generalized Advantage Estimation (GAE) using a learned value
     head, over the response tokens.
  4. PPO update: the clipped surrogate policy loss + a value function loss, for a few
     epochs over the same rollout batch (this is what makes PPO "proximal" -- it reuses
     the rollout data for multiple gradient steps while `ratio` clipping keeps each
     update conservative relative to the policy that generated the data).

Run:
    python labs/03_ppo_rlhf/scratch/train_ppo_scratch.py
"""

from __future__ import annotations

import copy
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.paths import run_dir  # noqa: E402
from common.rewards import SentimentReward  # noqa: E402
from common.utils import RunLogger, banner, decode_batch, set_seed  # noqa: E402

POLICY_MODEL_NAME = "lvwerra/gpt2-imdb"  # GPT-2 already fine-tuned on IMDB reviews
REWARD_MODEL_NAME = "lvwerra/distilbert-imdb"

# --- Task/rollout hyperparameters ---
BATCH_SIZE = 64
PROMPT_MIN_TOKENS = 4
PROMPT_MAX_TOKENS = 10
RESPONSE_TOKENS = 32
N_PPO_STEPS = 150

# --- PPO hyperparameters ---
PPO_EPOCHS = 4
CLIP_EPS = 0.2
VF_COEF = 0.1
ENT_COEF = 0.0
KL_BETA = 0.05  # try 0.0 (and watch reward hack into gibberish) -- see lab08
GAMMA = 1.0
GAE_LAMBDA = 0.95
LR = 1.41e-5
LOG_EVERY = 5


class PolicyWithValueHead(nn.Module):
    """A causal LM plus a scalar value head on top of its final hidden states. This is
    the "actor-critic" architecture PPO needs: the LM is the actor (policy over next
    tokens), the value head is the critic (estimates expected future reward from a
    given prefix), and they share the transformer trunk.
    """

    def __init__(self, model_name: str):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = getattr(self.lm.config, "hidden_size", None) or self.lm.config.n_embd
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.value_head.bias)
        nn.init.normal_(self.value_head.weight, std=1 / (hidden_size + 1) ** 0.5)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        values = self.value_head(out.hidden_states[-1]).squeeze(-1)  # (B, T)
        return out.logits, values

    def generate(self, *args, **kwargs):
        return self.lm.generate(*args, **kwargs)


def build_prompt_ids(tokenizer, n: int, seed: int = 0) -> list[list[int]]:
    """Truncated IMDB review prefixes, exactly like the original TRL sentiment demo:
    short, varied-length prompts so the model has to *generate* most of the content
    (as opposed to e.g. finishing an almost-complete sentence).
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("imdb", split="train").shuffle(seed=seed).select(range(n * 2))
    prompts = []
    for ex in ds:
        ids = tokenizer(ex["text"], truncation=True, max_length=PROMPT_MAX_TOKENS)["input_ids"]
        if len(ids) < PROMPT_MIN_TOKENS:
            continue
        plen = rng.randint(PROMPT_MIN_TOKENS, min(PROMPT_MAX_TOKENS, len(ids)))
        prompts.append(ids[:plen])
        if len(prompts) >= n:
            break
    return prompts


def left_pad(id_lists: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in id_lists)
    input_ids, attn_mask = [], []
    for ids in id_lists:
        pad_n = max_len - len(ids)
        input_ids.append([pad_id] * pad_n + ids)
        attn_mask.append([0] * pad_n + [1] * len(ids))
    return torch.tensor(input_ids, device=device), torch.tensor(attn_mask, device=device)


def get_response_logprobs_values(
    model: PolicyWithValueHead, full_ids: torch.Tensor, full_mask: torch.Tensor, prompt_len: int
):
    """Logits at position i predict the token at position i+1, so to score the response
    tokens (positions prompt_len .. prompt_len+R-1) we read logits/values from position
    (prompt_len - 1) .. (prompt_len + R - 2).
    """
    logits, values = model(full_ids, full_mask)
    response_len = full_ids.shape[1] - prompt_len
    shifted_logits = logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
    shifted_values = values[:, prompt_len - 1 : prompt_len - 1 + response_len]
    log_probs_all = F.log_softmax(shifted_logits, dim=-1)
    target_ids = full_ids[:, prompt_len : prompt_len + response_len]
    logprobs = torch.gather(log_probs_all, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs_all.exp() * log_probs_all).sum(-1)
    return logprobs, shifted_values, entropy


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float, lam: float):
    """Standard GAE(gamma, lambda) over the response tokens, treating the end of the
    generated response as the end of the episode (no bootstrapping past it).
    rewards/values/mask: (B, R).
    """
    B, R = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(R)):
        next_value = values[:, t + 1] if t + 1 < R else torch.zeros(B, device=rewards.device)
        next_mask = mask[:, t + 1] if t + 1 < R else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        last_gae = delta + gamma * lam * next_mask * last_gae
        advantages[:, t] = last_gae
    returns = advantages + values
    return advantages, returns


def main() -> None:
    set_seed(0)
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner("Lab 03 (scratch): PPO for RLHF on GPT-2-IMDB / sentiment reward")

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    policy = PolicyWithValueHead(POLICY_MODEL_NAME).to(device)
    ref_policy = copy.deepcopy(policy.lm).to(device).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    reward_model = SentimentReward(REWARD_MODEL_NAME, device=device, use_logit=True)

    optim = torch.optim.AdamW(policy.parameters(), lr=LR)
    logger = RunLogger(
        run_dir("lab03_ppo_scratch"), "ppo_scratch", config={"kl_beta": KL_BETA, "clip_eps": CLIP_EPS}
    )

    all_prompts = build_prompt_ids(tokenizer, n=BATCH_SIZE * N_PPO_STEPS, seed=0)

    for step in range(N_PPO_STEPS):
        # ---------------- 1. Rollout ----------------
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
        full_ids = gen_out
        full_mask = torch.cat([prompt_mask, torch.ones_like(response_ids)], dim=1)
        response_mask = torch.ones_like(response_ids, dtype=torch.float)  # fixed-length responses (min==max new tokens)

        # ---------------- 2. Score with the reward model ----------------
        full_texts = decode_batch(tokenizer, full_ids, full_mask)
        rewards_scalar = torch.tensor(reward_model(completions=full_texts), device=device)

        # ---------------- 3. Old policy / reference log-probs + values ----------------
        with torch.no_grad():
            old_logprobs, old_values, _ = get_response_logprobs_values(policy, full_ids, full_mask, prompt_len)
            ref_logits = ref_policy(input_ids=full_ids, attention_mask=full_mask).logits
            ref_shifted = ref_logits[:, prompt_len - 1 : prompt_len - 1 + RESPONSE_TOKENS, :]
            ref_logprobs_all = F.log_softmax(ref_shifted, dim=-1)
            ref_logprobs = torch.gather(ref_logprobs_all, 2, response_ids.unsqueeze(-1)).squeeze(-1)

        # Per-token KL penalty everywhere, plus the scalar reward only at each
        # sequence's last real token (here: always the last position, since responses
        # are fixed-length).
        per_token_kl = old_logprobs - ref_logprobs
        token_rewards = -KL_BETA * per_token_kl
        last_idx = response_mask.sum(dim=1).long() - 1
        token_rewards[torch.arange(BATCH_SIZE, device=device), last_idx] += rewards_scalar

        advantages, returns = compute_gae(token_rewards, old_values, response_mask, GAMMA, GAE_LAMBDA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---------------- 4. PPO update ----------------
        policy.train()
        for _ in range(PPO_EPOCHS):
            new_logprobs, new_values, entropy = get_response_logprobs_values(policy, full_ids, full_mask, prompt_len)
            ratio = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * ((new_values - returns) ** 2).mean()
            entropy_bonus = entropy.mean()

            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy_bonus

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                approx_kl = per_token_kl.mean().item()
                clip_frac = ((ratio - 1.0).abs() > CLIP_EPS).float().mean().item()
            logger.log(
                step,
                mean_reward=rewards_scalar.mean().item(),
                policy_loss=policy_loss.item(),
                value_loss=value_loss.item(),
                approx_kl=approx_kl,
                clip_frac=clip_frac,
            )
            print(f"    sample: {full_texts[0]!r}")

    logger.plot(["mean_reward", "approx_kl", "policy_loss", "value_loss"], title="PPO on GPT-2-IMDB")

    out_dir = run_dir("lab03_ppo_scratch") / "final_model"
    policy.lm.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved PPO-tuned policy to {out_dir}")
    print(
        "\nmean_reward should climb noticeably over training (the classifier's positive-class "
        "logit) while approx_kl stays modest -- if approx_kl blows up, the KL penalty (KL_BETA) "
        "is too weak for the learning rate/step count; if reward stalls near its starting value, "
        "it may be too strong. See lab08 for a deliberate demonstration of the failure mode."
    )


if __name__ == "__main__":
    main()
