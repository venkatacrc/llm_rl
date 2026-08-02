"""Lab 04 (scratch): Direct Preference Optimization, derived and coded by hand.

DPO's key insight: you don't need a separate reward model or an RL rollout loop at all.
Recall lab 02's Bradley-Terry reward-model loss:

    L_RM(phi) = -logsigmoid(r_phi(chosen) - r_phi(rejected))

DPO substitutes the *policy itself* (relative to a frozen reference) as the implicit
reward:

    r_theta(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )

Plugging this into the same Bradley-Terry loss gives a loss you can compute directly
from log-probabilities -- no reward model, no sampling/rollouts, no value function:

    L_DPO(theta) = -logsigmoid(
        beta * [ (logpi_theta(y_w|x) - logpi_ref(y_w|x))
               - (logpi_theta(y_l|x) - logpi_ref(y_l|x)) ]
    )

where y_w = chosen (winning) completion, y_l = rejected (losing) completion. This is
just supervised learning on log-prob *differences* -- you can compute it with two
forward passes (policy, reference) over (prompt, chosen) and (prompt, rejected), no
generation required during training at all. That's why DPO is so much simpler to
implement and more stable to train than PPO, at the cost of being strictly off-policy
(it never samples from the *current* policy during training, unlike PPO/GRPO).

Task/data: same spirit as lab 02's scratch reward model -- preference pairs built from
IMDB (chosen = positive-sentiment text, rejected = negative), but framed as completions
of a fixed generic prompt so DPO's prompt-conditional formulation applies cleanly.
Base model: `lvwerra/gpt2-imdb` (same starting point as lab 03's PPO run, so the two are
directly comparable).

Run:
    python labs/04_dpo_family/scratch/train_dpo_scratch.py
"""

from __future__ import annotations

import copy
import os
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.paths import run_dir  # noqa: E402
from common.utils import RunLogger, banner, set_seed  # noqa: E402

MODEL_NAME = "lvwerra/gpt2-imdb"
N_TRAIN_PAIRS = 4000
N_EVAL_PAIRS = 500
MAX_COMPLETION_TOKENS = 64
BATCH_SIZE = 16
N_EPOCHS = 1
LR = 1e-5
BETA = 0.1
LOG_EVERY = 20

PROMPT_POOL = [
    "The movie was",
    "This film is",
    "Overall, I thought the movie was",
    "My honest review:",
]


class PreferencePairDataset(Dataset):
    def __init__(self, prompts: list[str], pos_texts: list[str], neg_texts: list[str]):
        self.prompts, self.pos_texts, self.neg_texts = prompts, pos_texts, neg_texts

    def __len__(self):
        return len(self.pos_texts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "chosen": self.pos_texts[idx], "rejected": self.neg_texts[idx]}


def build_pairs(n_train: int, n_eval: int, seed: int = 0):
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=seed)
    pos = [ex["text"] for ex in ds if ex["label"] == 1]
    neg = [ex["text"] for ex in ds if ex["label"] == 0]
    n_total = n_train + n_eval
    pos, neg = pos[:n_total], neg[:n_total]
    rng.shuffle(neg)
    prompts = [rng.choice(PROMPT_POOL) for _ in range(n_total)]
    train = PreferencePairDataset(prompts[:n_train], pos[:n_train], neg[:n_train])
    eval_ = PreferencePairDataset(prompts[n_train:], pos[n_train:n_total], neg[n_train:n_total])
    return train, eval_


def make_collate(tokenizer):
    def encode_side(prompts, completions):
        """Build `input_ids = prompt + completion (+eos)`, right-padded, plus a
        `loss_mask` that is 1 only on completion tokens (so the DPO log-prob sum
        excludes the shared prompt, matching the math above).
        """
        input_ids_list, loss_masks = [], []
        for prompt, completion in zip(prompts, completions):
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, truncation=True, max_length=MAX_COMPLETION_TOKENS)[
                "input_ids"
            ] + [tokenizer.eos_token_id]
            ids = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            input_ids_list.append(ids)
            loss_masks.append(mask)
        max_len = max(len(x) for x in input_ids_list)
        input_ids, attn_mask, loss_mask = [], [], []
        for ids, mask in zip(input_ids_list, loss_masks):
            pad_n = max_len - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * pad_n)
            attn_mask.append([1] * len(ids) + [0] * pad_n)
            loss_mask.append(mask + [0] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn_mask),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
        }

    def collate(batch):
        prompts = [b["prompt"] for b in batch]
        chosen = encode_side(prompts, [b["chosen"] for b in batch])
        rejected = encode_side(prompts, [b["rejected"] for b in batch])
        return chosen, rejected

    return collate


def sequence_logps(model, encoded: dict) -> torch.Tensor:
    """Sum of per-token log-probs over the completion tokens only (see `loss_mask`)."""
    input_ids, attention_mask, loss_mask = encoded["input_ids"], encoded["attention_mask"], encoded["loss_mask"]
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logps = F.log_softmax(logits, dim=-1)
    token_logps = torch.gather(logps, 2, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * loss_mask[:, 1:]).sum(-1)


def dpo_loss(policy, ref, chosen, rejected, beta: float):
    policy_chosen_logps = sequence_logps(policy, chosen)
    policy_rejected_logps = sequence_logps(policy, rejected)
    with torch.no_grad():
        ref_chosen_logps = sequence_logps(ref, chosen)
        ref_rejected_logps = sequence_logps(ref, rejected)

    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_reward - rejected_reward
    loss = -F.logsigmoid(logits).mean()
    accuracy = (logits > 0).float().mean()
    return loss, accuracy, chosen_reward.mean(), rejected_reward.mean()


def main() -> None:
    set_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner(f"Lab 04 (scratch): DPO on {MODEL_NAME} / IMDB preference pairs")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    ref = copy.deepcopy(policy).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    train_ds, eval_ds = build_pairs(N_TRAIN_PAIRS, N_EVAL_PAIRS)
    collate = make_collate(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    optim = torch.optim.AdamW(policy.parameters(), lr=LR)
    logger = RunLogger(run_dir("lab04_dpo_scratch"), "dpo_scratch", config={"beta": BETA, "lr": LR})

    def to_device(batch):
        chosen, rejected = batch
        chosen = {k: v.to(device) for k, v in chosen.items()}
        rejected = {k: v.to(device) for k, v in rejected.items()}
        return chosen, rejected

    step = 0
    for epoch in range(N_EPOCHS):
        for batch in train_loader:
            chosen, rejected = to_device(batch)
            loss, acc, r_chosen, r_rejected = dpo_loss(policy, ref, chosen, rejected, BETA)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            optim.zero_grad()

            if step % LOG_EVERY == 0:
                logger.log(
                    step,
                    loss=loss.item(),
                    accuracy=acc.item(),
                    chosen_reward=r_chosen.item(),
                    rejected_reward=r_rejected.item(),
                    margin=(r_chosen - r_rejected).item(),
                )
            step += 1

    logger.plot(["loss", "accuracy", "margin"], title="DPO training (GPT-2-IMDB)")

    print("\nEval on held-out pairs:")
    policy.eval()
    total_acc, n = 0.0, 0
    with torch.no_grad():
        for batch in eval_loader:
            chosen, rejected = to_device(batch)
            _, acc, _, _ = dpo_loss(policy, ref, chosen, rejected, BETA)
            total_acc += acc.item() * len(chosen["input_ids"])
            n += len(chosen["input_ids"])
    print(f"  held-out preference accuracy: {total_acc / n:.3f}")

    out_dir = run_dir("lab04_dpo_scratch") / "final_model"
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved DPO-tuned policy to {out_dir}")
    print(
        "\nCompare this against lab 03's PPO-tuned GPT-2-IMDB: both start from the same base "
        "model and the same underlying preference signal (positive > negative sentiment), but "
        "DPO got there without ever sampling from its own policy during training."
    )


if __name__ == "__main__":
    main()
