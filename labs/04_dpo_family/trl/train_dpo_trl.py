"""Lab 04 (at scale): Direct Preference Optimization with TRL's `DPOTrainer`.

Same loss as `scratch/train_dpo_scratch.py`, on Qwen2.5-0.5B starting from the lab-01
SFT checkpoint, trained on the exact same preference data lab 02 used for reward
modeling (`trl-lib/tldr-preference`) -- so you can directly compare "reward model +
PPO" (lab 03) against "skip the reward model, optimize the policy directly" (this lab).

Single GPU:
    python labs/04_dpo_family/trl/train_dpo_trl.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.paths import SFT_MODEL_NAME, checkpoint_dir, run_dir  # noqa: E402
from common.utils import banner, set_seed  # noqa: E402

FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "trl-lib/tldr-preference"


def resolve_base_model() -> str:
    sft_dir = checkpoint_dir(SFT_MODEL_NAME, create=False)
    if sft_dir.exists() and any(sft_dir.iterdir()):
        return str(sft_dir)
    print(f"[warn] no lab-01 SFT checkpoint at {sft_dir}; falling back to {FALLBACK_MODEL}.")
    return FALLBACK_MODEL


def main() -> None:
    set_seed(0)
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    banner(f"Lab 04 (TRL): DPO on {DATASET_NAME}")

    base_model = resolve_base_model()
    train_dataset = load_dataset(DATASET_NAME, split="train")
    eval_dataset = load_dataset(DATASET_NAME, split="validation").select(range(500))

    out_dir = checkpoint_dir("dpo_qwen2.5-0.5b_tldr")
    args = DPOConfig(
        output_dir=str(out_dir),
        logging_dir=str(run_dir("lab04_dpo_trl")),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,  # DPO is sensitive to LR; this is the standard "small" DPO LR
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=1,
        beta=0.1,
        bf16=True,
        max_length=640,
        max_prompt_length=512,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        report_to=[r for r in ["wandb"] if os.environ.get("WANDB_PROJECT")] or ["tensorboard"],
        run_name="lab04_dpo_trl",
    )

    trainer = DPOTrainer(
        model=base_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    trainer.processing_class.save_pretrained(str(out_dir))
    print(f"\nSaved DPO model to {out_dir}")


if __name__ == "__main__":
    main()
