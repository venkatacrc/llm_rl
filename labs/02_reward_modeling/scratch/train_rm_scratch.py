"""Lab 02 (scratch): reward modeling from the Bradley-Terry model, by hand.

The reward model (RM) is the piece PPO (lab 03) optimizes against. It's trained on
*pairwise preferences*: given a prompt/context and two candidate completions, humans (or,
for RLVR, a rule) say which one is better. We don't have access to an absolute quality
score -- only "A is better than B" -- so we need a model of how preferences arise from
a latent scalar reward. That's the Bradley-Terry model:

    P(A preferred over B) = sigmoid(r(A) - r(B))

which gives the (negative log-likelihood) training loss for a reward model r_phi with
a scalar output head:

    L(phi) = -log( sigmoid( r_phi(chosen) - r_phi(rejected) ) )
           = -logsigmoid(r_phi(chosen) - r_phi(rejected))

Minimizing this pushes the model to assign a higher score to "chosen" than "rejected"
-- note the loss only cares about the *gap*, not the absolute scale, which is why
reward model outputs are typically normalized/centered before being used for RL.

Task/data: to keep this self-contained and fast, we build preference pairs from the
IMDB sentiment dataset itself: `chosen` = a review labeled positive, `rejected` = a
review labeled negative. A well-trained RM should learn to assign high reward to
positive-sentiment text -- which we can directly sanity-check against the ground-truth
IMDB labels held out from training. (Lab 03 uses a pretrained sentiment classifier as
its scratch PPO reward for speed, but this RM -- trained the "real" RLHF way, from
pairwise preferences rather than a labeled classifier -- would work as a drop-in
replacement; see lab02 README "things to try".)

Run:
    python labs/02_reward_modeling/scratch/train_rm_scratch.py
"""

from __future__ import annotations

import os
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.utils import RunLogger, banner, set_seed  # noqa: E402
from common.paths import run_dir  # noqa: E402

MODEL_NAME = "gpt2"
N_TRAIN_PAIRS = 4000
N_EVAL_PAIRS = 500
MAX_LENGTH = 256
BATCH_SIZE = 16
N_EPOCHS = 1
LR = 1e-5
LOG_EVERY = 20


class PreferencePairDataset(Dataset):
    def __init__(self, pos_texts: list[str], neg_texts: list[str]):
        assert len(pos_texts) == len(neg_texts)
        self.pos_texts = pos_texts
        self.neg_texts = neg_texts

    def __len__(self):
        return len(self.pos_texts)

    def __getitem__(self, idx):
        return {"chosen": self.pos_texts[idx], "rejected": self.neg_texts[idx]}


def make_collate(tokenizer):
    def collate(batch):
        chosen = tokenizer(
            [b["chosen"] for b in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        rejected = tokenizer(
            [b["rejected"] for b in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return chosen, rejected

    return collate


def build_pairs(n_train: int, n_eval: int, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=seed)
    pos = [ex["text"] for ex in ds if ex["label"] == 1]
    neg = [ex["text"] for ex in ds if ex["label"] == 0]
    n_total = n_train + n_eval
    pos, neg = pos[:n_total], neg[:n_total]
    random.Random(seed).shuffle(neg)  # decouple pos[i]/neg[i] pairing from dataset order
    train = PreferencePairDataset(pos[:n_train], neg[:n_train])
    eval_ = PreferencePairDataset(pos[n_train:n_total], neg[n_train:n_total])
    return train, eval_


def reward_scores(model, tokenized_batch, device) -> torch.Tensor:
    tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
    logits = model(**tokenized_batch).logits  # (B, 1), GPT2ForSequenceClassification pools the last real token
    return logits.squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    for chosen, rejected in loader:
        r_chosen = reward_scores(model, chosen, device)
        r_rejected = reward_scores(model, rejected, device)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        total_loss += loss.item() * len(r_chosen)
        total_correct += (r_chosen > r_rejected).sum().item()
        n += len(r_chosen)
    model.train()
    return total_loss / n, total_correct / n


def main() -> None:
    set_seed(0)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner(f"Lab 02 (scratch): Bradley-Terry reward model on {MODEL_NAME} / IMDB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_ds, eval_ds = build_pairs(N_TRAIN_PAIRS, N_EVAL_PAIRS)
    collate = make_collate(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    logger = RunLogger(run_dir("lab02_rm_scratch"), "rm_scratch", config={"model": MODEL_NAME, "lr": LR})

    print("\nBefore training:")
    loss0, acc0 = evaluate(model, eval_loader, device)
    print(f"  eval loss={loss0:.4f}  pairwise accuracy={acc0:.3f} (should be ~0.5, untrained)")

    step = 0
    for epoch in range(N_EPOCHS):
        for chosen, rejected in train_loader:
            r_chosen = reward_scores(model, chosen, device)
            r_rejected = reward_scores(model, rejected, device)
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()

            if step % LOG_EVERY == 0:
                acc = (r_chosen > r_rejected).float().mean().item()
                margin = (r_chosen - r_rejected).mean().item()
                logger.log(step, loss=loss.item(), train_pairwise_acc=acc, mean_margin=margin)
            step += 1

    logger.plot(["loss", "train_pairwise_acc", "mean_margin"], title="Reward model training (GPT-2 / IMDB)")

    print("\nAfter training:")
    loss1, acc1 = evaluate(model, eval_loader, device)
    print(f"  eval loss={loss1:.4f}  pairwise accuracy={acc1:.3f} (held-out positive vs negative reviews)")

    out_dir = run_dir("lab02_rm_scratch") / "final_model"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved reward model to {out_dir}")
    print(
        "\nA pairwise accuracy well above 0.5 (typically >0.85 with this much data) means the "
        "RM has learned 'positive review text scores higher than negative review text' purely "
        "from *relative* preferences -- it never saw the word 'positive' or a sentiment label."
    )


if __name__ == "__main__":
    main()
