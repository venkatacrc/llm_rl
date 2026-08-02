"""Lab 04 (at scale): Kahneman-Tversky Optimization with TRL's `KTOTrainer`.

KTO relaxes DPO's requirement of *paired* preferences (needing both a chosen and a
rejected response to the *same* prompt) down to independent binary "thumbs up / thumbs
down" labels per (prompt, completion) -- often much cheaper to collect in practice, and
motivated by prospect theory (humans weigh losses more heavily than equivalent gains,
which shows up as an asymmetric loss around a reference point rather than DPO's
symmetric logistic loss).

We reuse the same `trl-lib/tldr-preference` (chosen/rejected pairs) dataset as lab 04's
DPO run and lab 02's reward model -- `KTOTrainer` auto-converts paired preference data
into KTO's unpaired (prompt, completion, label) format internally, so this is an
apples-to-apples comparison of *loss function*, not *data*.

Single GPU:
    python labs/04_dpo_family/trl/train_kto_trl.py
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
    from trl import KTOConfig, KTOTrainer

    banner(f"Lab 04 (TRL): KTO on {DATASET_NAME} (paired -> auto-converted to unpaired)")

    base_model = resolve_base_model()
    train_dataset = load_dataset(DATASET_NAME, split="train")
    eval_dataset = load_dataset(DATASET_NAME, split="validation").select(range(500))

    out_dir = checkpoint_dir("kto_qwen2.5-0.5b_tldr")
    args = KTOConfig(
        output_dir=str(out_dir),
        logging_dir=str(run_dir("lab04_kto_trl")),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=1,
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        bf16=True,
        max_length=640,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        report_to=[r for r in ["wandb"] if os.environ.get("WANDB_PROJECT")] or ["tensorboard"],
        run_name="lab04_kto_trl",
    )

    trainer = KTOTrainer(
        model=base_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    trainer.processing_class.save_pretrained(str(out_dir))
    print(f"\nSaved KTO model to {out_dir}")


if __name__ == "__main__":
    main()
