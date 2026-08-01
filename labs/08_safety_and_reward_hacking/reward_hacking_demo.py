"""Lab 08, part 1: reward hacking / Goodhart's law, deliberately induced.

"When a measure becomes a target, it ceases to be a good measure." We optimize a
policy against a deliberately naive **proxy** reward -- `length_reward` from
`common/rewards.py`, which just rewards longer completions -- while *monitoring* (but
never training against) two independent signals that stand in for "what we actually
wanted": the sentiment classifier reward from lab 03, and a simple repetition-rate
metric. The point: a policy trained with no/weak KL constraint against a
misspecified/gameable reward will learn to maximize *that number*, not the underlying
goal you actually had in mind -- often visibly, by degrading into repetitive filler.

This uses the simplest possible RL update (REINFORCE with a batch-mean baseline, no
critic, no clipping) precisely so the *reward hacking phenomenon* is the only
interesting thing happening -- see labs 03/05/06 for the algorithms themselves.

Run:
    python labs/08_safety_and_reward_hacking/reward_hacking_demo.py                 # KL_BETA=0.0: watch it hack
    python labs/08_safety_and_reward_hacking/reward_hacking_demo.py --kl-beta 0.2   # constrained: compare
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.paths import run_dir  # noqa: E402
from common.rewards import SentimentReward, length_reward  # noqa: E402
from common.utils import RunLogger, banner, decode_batch, set_seed  # noqa: E402

POLICY_MODEL_NAME = "lvwerra/gpt2-imdb"
BATCH_SIZE = 64
PROMPT_MIN_TOKENS = 4
PROMPT_MAX_TOKENS = 10
RESPONSE_TOKENS = 48
N_STEPS = 150
LR = 1.41e-5
LOG_EVERY = 5


def build_prompt_ids(tokenizer, n: int, seed: int = 0) -> list[list[int]]:
    """Short IMDB review prefixes -- same recipe as lab 03's scratch PPO script."""
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


def repetition_rate(texts: list[str], n: int = 3) -> float:
    """Fraction of n-grams that are *repeats* of an earlier n-gram in the same text
    (0 = no repetition, higher = more degenerate/repetitive). A classic, cheap tell for
    reward-hacked text that's exploiting a length or keyword-count proxy.
    """
    rates = []
    for t in texts:
        words = t.split()
        if len(words) < n + 1:
            rates.append(0.0)
            continue
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        seen, repeats = set(), 0
        for g in ngrams:
            if g in seen:
                repeats += 1
            seen.add(g)
        rates.append(repeats / len(ngrams))
    return sum(rates) / len(rates)


def get_response_logprobs(model, full_ids, full_mask, prompt_len, response_len):
    logits = model(input_ids=full_ids, attention_mask=full_mask).logits
    shifted = logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
    log_probs_all = F.log_softmax(shifted, dim=-1)
    target = full_ids[:, prompt_len : prompt_len + response_len]
    return torch.gather(log_probs_all, 2, target.unsqueeze(-1)).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kl-beta", type=float, default=0.0, help="0.0 = no KL constraint -> full reward hacking")
    args = parser.parse_args()

    set_seed(0)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner(f"Lab 08: reward hacking demo (proxy=length, kl_beta={args.kl_beta})")

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(device)
    ref_policy = copy.deepcopy(policy).to(device).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    sentiment_reward = SentimentReward(device=device)  # "true" quality signal -- NEVER trained against

    optim = torch.optim.AdamW(policy.parameters(), lr=LR)
    logger = RunLogger(
        run_dir(f"lab08_reward_hacking_beta{args.kl_beta}"),
        "reward_hacking",
        config={"kl_beta": args.kl_beta, "proxy": "length"},
    )

    all_prompts = build_prompt_ids(tokenizer, n=BATCH_SIZE * N_STEPS, seed=0)

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
        response_texts = decode_batch(tokenizer, response_ids)

        # --- The optimized (proxy) reward ---
        proxy_rewards = torch.tensor(length_reward(completions=full_texts), device=device)
        # --- Monitoring-only signals, never backpropagated through ---
        true_rewards = torch.tensor(sentiment_reward(completions=full_texts), device=device)
        rep_rate = repetition_rate(response_texts)

        with torch.no_grad():
            old_logprobs = get_response_logprobs(policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)
            ref_logprobs = get_response_logprobs(ref_policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)
        per_token_kl = old_logprobs - ref_logprobs

        advantages = proxy_rewards - proxy_rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)

        policy.train()
        new_logprobs = get_response_logprobs(policy, gen_out, full_mask, prompt_len, RESPONSE_TOKENS)
        seq_logprob = new_logprobs.sum(dim=1)
        pg_loss = -(advantages.detach() * seq_logprob).mean()
        kl_loss = args.kl_beta * per_token_kl.mean()
        loss = pg_loss + kl_loss

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()

        if step % LOG_EVERY == 0:
            logger.log(
                step,
                proxy_reward=proxy_rewards.mean().item(),
                true_reward_sentiment=true_rewards.mean().item(),
                repetition_rate=rep_rate,
                approx_kl=per_token_kl.mean().item(),
            )
            print(f"    sample: {response_texts[0]!r}")

    logger.plot(
        ["proxy_reward", "true_reward_sentiment", "repetition_rate", "approx_kl"],
        title=f"Reward hacking demo (kl_beta={args.kl_beta})",
    )
    print(
        "\nIf kl_beta was left at 0.0: proxy_reward should climb to its ceiling while "
        "repetition_rate rises and true_reward_sentiment stays flat or drops -- the model found "
        "the cheapest way to be 'long', not a better way to be a positive review. Re-run with "
        "--kl-beta 0.2+ and compare: the KL penalty should visibly suppress the hack."
    )


if __name__ == "__main__":
    main()
