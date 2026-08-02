"""Lab 05 (at scale): GRPO + RLVR with TRL's `GRPOTrainer`, DeepSeek-R1-Zero-style.

Same algorithm as `scratch/train_grpo_scratch.py` (group-sampled rollouts,
group-normalized advantage, no critic, loss-level KL term), on
`Qwen/Qwen2.5-Math-1.5B` with GSM8K, using vLLM to make the "sample 8 completions per
prompt" rollout step fast. No reward model anywhere in this pipeline -- `gsm8k_reward`
(see `../env/math_reward.py`) is a pure rule: parse the model's final answer, compare
to the ground truth.

Uses vLLM in **colocate** mode (shares the training GPU(s) with generation, sleeping
between rollouts to free VRAM) per current TRL guidance for single-node GRPO -- no
separate vLLM server process needed.

Single GPU:
    python labs/05_grpo_rlvr_reasoning/trl/train_grpo_trl.py

Multi-GPU (recommended -- GRPO wants many groups in flight; use DeepSpeed ZeRO-3 so
optimizer states are sharded, leaving more headroom for the colocated vLLM engine):
    accelerate launch --config_file configs/accelerate/deepspeed_zero3_8gpu.yaml \\
        labs/05_grpo_rlvr_reasoning/trl/train_grpo_trl.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.paths import checkpoint_dir, run_dir  # noqa: E402
from common.utils import banner, set_seed  # noqa: E402
from env.math_reward import gsm8k_reward, load_gsm8k  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def main() -> None:
    set_seed(0)
    from trl import GRPOConfig, GRPOTrainer

    banner(f"Lab 05 (TRL): GRPO + RLVR on {MODEL_NAME} / GSM8K")

    train_dataset = load_gsm8k("train")
    eval_dataset = load_gsm8k("test").select(range(200))

    out_dir = checkpoint_dir("grpo_qwen2.5-math-1.5b_gsm8k")
    args = GRPOConfig(
        output_dir=str(out_dir),
        logging_dir=str(run_dir("lab05_grpo_trl")),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,  # the GRPO "group size" G
        max_completion_length=512,
        learning_rate=1e-6,
        beta=0.02,  # loss-level KL coefficient, same role as KL_BETA in the scratch script
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
        run_name="lab05_grpo_trl",
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=args,
        reward_funcs=gsm8k_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(str(out_dir))

    print(f"\nSaved GRPO-tuned model to {out_dir}")
    print("This recipe is what lab 06 (GSPO variant) and lab 07 (capstone scale-up) build on.")


if __name__ == "__main__":
    main()
