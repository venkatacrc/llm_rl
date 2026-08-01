"""Lab 02 (at scale): reward modeling with TRL's `RewardTrainer`.

Same Bradley-Terry objective as `scratch/train_rm_scratch.py`
(`-logsigmoid(r_chosen - r_rejected)`), but TRL handles tokenization/padding/collation
for you, and we start from the lab-01 SFT checkpoint (standard practice: the RM is
usually initialized from the same SFT policy it will later judge, so it starts out
"speaking the same language").

Dataset: `trl-lib/tldr-preference` (prompt + chosen/rejected TL;DR summaries) -- the
preference-pair version of the summarization task lab 01 fine-tuned on.

Single GPU:
    python labs/02_reward_modeling/trl/train_rm_trl.py

Multi-GPU:
    accelerate launch --config_file configs/accelerate/ddp_8gpu.yaml \\
        labs/02_reward_modeling/trl/train_rm_trl.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.paths import REWARD_MODEL_NAME, SFT_MODEL_NAME, checkpoint_dir, run_dir  # noqa: E402
from common.utils import banner, set_seed  # noqa: E402

BASE_MODEL_FALLBACK = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "trl-lib/tldr-preference"


def resolve_base_model() -> str:
    sft_dir = checkpoint_dir(SFT_MODEL_NAME, create=False)
    if sft_dir.exists() and any(sft_dir.iterdir()):
        print(f"Starting reward model from lab-01 SFT checkpoint: {sft_dir}")
        return str(sft_dir)
    print(f"No lab-01 SFT checkpoint found at {sft_dir}; falling back to {BASE_MODEL_FALLBACK}. "
          f"Run labs/01_sft_warmup/trl/train_sft_trl.py first for the intended pipeline.")
    return BASE_MODEL_FALLBACK


def main() -> None:
    set_seed(0)
    from datasets import load_dataset
    from trl import RewardConfig, RewardTrainer

    banner(f"Lab 02 (TRL): reward modeling on {DATASET_NAME}")

    base_model = resolve_base_model()
    train_dataset = load_dataset(DATASET_NAME, split="train")
    eval_dataset = load_dataset(DATASET_NAME, split="validation").select(range(500))

    out_dir = checkpoint_dir(REWARD_MODEL_NAME)
    args = RewardConfig(
        output_dir=str(out_dir),
        logging_dir=str(run_dir("lab02_rm_trl")),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=1,
        bf16=True,
        max_length=640,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        report_to=[r for r in ["wandb"] if os.environ.get("WANDB_PROJECT")] or ["tensorboard"],
        run_name="lab02_rm_trl",
        center_rewards_coefficient=0.01,  # keeps mean reward near 0, standard trick for RM stability
    )

    trainer = RewardTrainer(
        model=base_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal eval metrics: {metrics}")

    trainer.save_model(str(out_dir))
    trainer.processing_class.save_pretrained(str(out_dir))
    print(f"\nSaved reward model to {out_dir}")
    print("This checkpoint is what lab 03's PPO script and lab 08's reward-hacking demo load.")


if __name__ == "__main__":
    main()
