"""Lab 07 (capstone): the lab-05 GRPO/RLVR recipe, scaled to a 7-8B model across the
full 8xB200 node.

Identical algorithm and reward (`env/math_reward.py`'s GSM8K verifier) to
`labs/05_grpo_rlvr_reasoning/trl/train_grpo_trl.py` -- the only thing that changes here
is scale: a larger base model, sharded with FSDP across all 8 GPUs (each GPU holding a
shard of the policy + reference model), with vLLM colocated for rollout generation.

This is deliberately the *same script structure* as lab 05's GRPO run so the diff
between "toy scale" and "real scale" post-training is as small and legible as possible
-- scaling up RL post-training is mostly an infra problem, not an algorithm problem.

Run (needs all 8 GPUs; FSDP shards the ~7B-parameter policy + reference model across
them -- each of the 183GB B200s could actually hold an unsharded 7B model on its own,
but FSDP is what you'd need the moment you point this at a 30B+ model, so the capstone
exercises it deliberately):

    accelerate launch --config_file configs/accelerate/fsdp_8gpu.yaml \\
        labs/07_capstone_scale_up/train_grpo_capstone.py

Alternative (DeepSpeed ZeRO-3 -- often plays more smoothly with colocated vLLM since
optimizer-state offload frees more VRAM for the inference engine; try this first if
FSDP + vLLM colocate misbehaves on your TRL/vLLM version combination):

    accelerate launch --config_file configs/accelerate/deepspeed_zero3_8gpu.yaml \\
        labs/07_capstone_scale_up/train_grpo_capstone.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "05_grpo_rlvr_reasoning"))
from common.paths import checkpoint_dir, run_dir  # noqa: E402
from common.utils import banner, set_seed  # noqa: E402
from env.math_reward import gsm8k_reward, load_gsm8k  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    from trl import GRPOConfig, GRPOTrainer

    banner(f"Lab 07 (capstone): GRPO + RLVR on {args.model} across {os.environ.get('WORLD_SIZE', 1)} GPU(s)")

    train_dataset = load_gsm8k("train")
    eval_dataset = load_gsm8k("test").select(range(200))

    run_name = f"lab07_capstone_{args.model.split('/')[-1]}"
    out_dir = checkpoint_dir(run_name)
    training_args = GRPOConfig(
        output_dir=str(out_dir),
        logging_dir=str(run_dir(run_name)),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=768,
        learning_rate=1e-6,
        beta=0.02,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        report_to=[r for r in ["wandb"] if os.environ.get("WANDB_PROJECT")] or ["tensorboard"],
        run_name=run_name,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        reward_funcs=gsm8k_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(str(out_dir))

    print(f"\nSaved capstone model to {out_dir}")
    print(
        "Compare GSM8K accuracy (eval reward) and wall-clock/step against lab 05's 1.5B run -- "
        "see benchmark_scaling.py for a controlled 1/2/4/8-GPU throughput comparison."
    )


if __name__ == "__main__":
    main()
