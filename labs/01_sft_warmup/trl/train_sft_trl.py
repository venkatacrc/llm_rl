"""Lab 01 (at scale): supervised fine-tuning with TRL's `SFTTrainer`.

Same idea as `scratch/train_sft_scratch.py` (prompt-masked next-token prediction), but
on a real instruction-following base model (Qwen2.5-0.5B) using TRL, which handles
label masking, packing, and chat templates for you. This produces the SFT checkpoint
that labs 02-06's TRL scripts start from -- i.e. this is π_ref / π_init for the rest of
the "at scale" curriculum.

Single GPU:
    python labs/01_sft_warmup/trl/train_sft_trl.py

Data-parallel across all 8 GPUs (bigger effective batch size, same result faster):
    accelerate launch --config_file configs/accelerate/ddp_8gpu.yaml \\
        labs/01_sft_warmup/trl/train_sft_trl.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.paths import SFT_MODEL_NAME, checkpoint_dir, run_dir  # noqa: E402
from common.utils import banner, set_seed  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "trl-lib/tldr"


def main() -> None:
    set_seed(0)
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    banner(f"Lab 01 (TRL): SFT on {MODEL_NAME} / {DATASET_NAME}")

    train_dataset = load_dataset(DATASET_NAME, split="train")
    eval_dataset = load_dataset(DATASET_NAME, split="validation").select(range(200))

    out_dir = checkpoint_dir(SFT_MODEL_NAME)
    args = SFTConfig(
        output_dir=str(out_dir),
        logging_dir=str(run_dir("lab01_sft_trl")),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=1,
        bf16=True,
        max_length=640,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        report_to=[r for r in ["wandb"] if os.environ.get("WANDB_PROJECT")] or ["tensorboard"],
        run_name="lab01_sft_trl",
    )

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    trainer.processing_class.save_pretrained(str(out_dir))

    print(f"\nSaved SFT model to {out_dir}")
    print("This checkpoint is what labs 02-06's TRL scripts load as their starting policy.")


if __name__ == "__main__":
    main()
