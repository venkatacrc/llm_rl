"""Lab 06 (at scale): RLOO with TRL's `RLOOTrainer`.

Same REINFORCE-with-leave-one-out-baseline idea as `scratch/train_rloo_scratch.py`, on
the same GSM8K/RLVR task lab 05 used for GRPO (`Qwen/Qwen2.5-Math-1.5B`), so you can
directly compare training curves between the two algorithms on identical data.

Single GPU:
    python labs/06_rloo_and_gspo/trl/train_rloo_trl.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "05_grpo_rlvr_reasoning"))
from common.paths import checkpoint_dir, run_dir  # noqa: E402
from common.utils import banner, set_seed  # noqa: E402
from env.math_reward import gsm8k_reward, load_gsm8k  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def main() -> None:
    set_seed(0)
    from trl import RLOOConfig, RLOOTrainer

    banner(f"Lab 06 (TRL): RLOO + RLVR on {MODEL_NAME} / GSM8K")

    train_dataset = load_gsm8k("train")
    eval_dataset = load_gsm8k("test").select(range(200))

    out_dir = checkpoint_dir("rloo_qwen2.5-math-1.5b_gsm8k")
    args = RLOOConfig(
        output_dir=str(out_dir),
        logging_dir=str(run_dir("lab06_rloo_trl")),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,  # group size, same role as GROUP_SIZE in lab 05/06 scratch scripts
        max_completion_length=512,
        learning_rate=1e-6,
        beta=0.02,
        num_train_epochs=1,
        bf16=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        report_to=[r for r in ["wandb"] if os.environ.get("WANDB_PROJECT")] or ["tensorboard"],
        run_name="lab06_rloo_trl",
    )

    trainer = RLOOTrainer(
        model=MODEL_NAME,
        args=args,
        reward_funcs=gsm8k_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(str(out_dir))

    print(f"\nSaved RLOO-tuned model to {out_dir}")
    print("Compare its training curves (runs/lab06_rloo_trl) against lab 05's GRPO run (runs/lab05_grpo_trl).")


if __name__ == "__main__":
    main()
