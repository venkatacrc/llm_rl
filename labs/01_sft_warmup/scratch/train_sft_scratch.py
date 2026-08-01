"""Lab 01 (scratch): supervised fine-tuning, the hand-written version.

Why this comes before any RL: every RLHF/RLVR pipeline starts from a model that can
already follow instructions/format reasonably well (the "SFT policy" or π_ref). RL then
*shifts* this policy towards higher reward without teaching it to produce text from
scratch. This script is the whole pipeline everything else builds on:

    tokenize(prompt + completion) -> mask the loss on prompt tokens -> next-token
    cross-entropy on the rest -> AdamW step.

There is no RL algorithm here yet -- this is just "the training loop", written by hand
so the mechanics (padding, label masking, generation for eyeballing progress) are
obvious before layering PPO/GRPO/DPO on top of it in later labs.

Model: GPT-2 (124M). Task: Reddit TL;DR summarization (`trl-lib/tldr`), the same
dataset lab 02/03/04/06 reuse, so you can directly compare "the base SFT model" against
what each RL algorithm does to it.

Run:
    python labs/01_sft_warmup/scratch/train_sft_scratch.py
"""

from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.paths import run_dir  # noqa: E402
from common.utils import RunLogger, banner, set_seed  # noqa: E402

MODEL_NAME = "gpt2"
N_TRAIN_EXAMPLES = 4000
N_EVAL_EXAMPLES = 8
MAX_PROMPT_TOKENS = 256
MAX_COMPLETION_TOKENS = 64
BATCH_SIZE = 16
N_EPOCHS = 1
LR = 5e-5
LOG_EVERY = 20
GENERATE_EVERY = 200


class PromptCompletionDataset(Dataset):
    """Tokenizes (prompt, completion) pairs and produces `labels` with the prompt
    portion masked to -100, so the loss is only computed on completion tokens -- the
    single most important detail in SFT that's easy to get wrong.
    """

    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tok = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        completion = ex["completion"] + self.tok.eos_token

        prompt_ids = self.tok(prompt, truncation=True, max_length=MAX_PROMPT_TOKENS)["input_ids"]
        completion_ids = self.tok(completion, truncation=True, max_length=MAX_COMPLETION_TOKENS)["input_ids"]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn_mask = [], [], []
    for b in batch:
        pad_n = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_n)
        labels.append(b["labels"] + [-100] * pad_n)
        attn_mask.append([1] * len(b["input_ids"]) + [0] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn_mask),
    }


@torch.no_grad()
def show_samples(model, tokenizer, prompts, device):
    model.eval()
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS).input_ids.to(device)
        out = model.generate(
            ids, max_new_tokens=MAX_COMPLETION_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
        completion = tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        print(f"  PROMPT (truncated): ...{p[-200:]!r}")
        print(f"  MODEL SUMMARY     : {completion!r}\n")
    model.train()


def main() -> None:
    set_seed(0)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    banner(f"Lab 01 (scratch): SFT on {MODEL_NAME} / TL;DR summarization")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    ds = load_dataset("trl-lib/tldr", split="train")
    train_examples = ds.select(range(N_TRAIN_EXAMPLES))
    eval_prompts = [ds[i]["prompt"] for i in range(N_TRAIN_EXAMPLES, N_TRAIN_EXAMPLES + N_EVAL_EXAMPLES)]

    print("\nBefore any training, GPT-2's idea of 'summarizing' (mostly rambling on):")
    show_samples(model, tokenizer, eval_prompts[:2], device)

    train_ds = PromptCompletionDataset(train_examples, tokenizer)
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    n_steps = len(loader) * N_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_steps)

    logger = RunLogger(run_dir("lab01_sft_scratch"), "sft_scratch", config={"model": MODEL_NAME, "lr": LR})

    step = 0
    for epoch in range(N_EPOCHS):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad()

            if step % LOG_EVERY == 0:
                logger.log(step, loss=loss.item(), lr=scheduler.get_last_lr()[0])
            if step % GENERATE_EVERY == 0 and step > 0:
                print(f"\n-- samples at step {step} --")
                show_samples(model, tokenizer, eval_prompts[:2], device)
            step += 1

    logger.plot(["loss"], title="SFT loss (GPT-2 / TL;DR)")

    print("\nFinal samples:")
    show_samples(model, tokenizer, eval_prompts, device)

    out_dir = run_dir("lab01_sft_scratch") / "final_model"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved fine-tuned GPT-2 to {out_dir}")
    print(
        "\nNote: this is a small (4k example, 1 epoch) demo run -- expect noticeably "
        "shorter/more on-topic completions than the pretrained baseline, not a polished "
        "summarizer. See lab01/README.md for things to try (more data/epochs, LR sweep)."
    )


if __name__ == "__main__":
    main()
