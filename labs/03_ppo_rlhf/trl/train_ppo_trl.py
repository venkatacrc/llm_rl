"""Lab 03 (at scale): PPO for RLHF with TRL's `PPOTrainer`.

Same algorithm as `scratch/train_ppo_scratch.py` (rollout -> reward model + KL penalty
-> GAE -> clipped surrogate PPO update), but on Qwen2.5-0.5B using the reward model
trained in lab 02 (`checkpoints/reward_model_qwen2.5-0.5b_tldr`), starting from the lab
01 SFT checkpoint. TRL handles rollout batching, the value model initialization, and
(if you enable it) vLLM-accelerated generation for you.

Note: as of TRL v1, `PPOTrainer`/`PPOConfig` live under `trl.experimental.ppo` -- GRPO
(lab 05) has become the more common recipe for LLM RL in practice, but PPO is still the
clearest way to see the "reward model + KL penalty + clipped surrogate + critic"
InstructGPT recipe in its original form.

Single GPU:
    python labs/03_ppo_rlhf/trl/train_ppo_trl.py

Multi-GPU (DeepSpeed ZeRO-3 -- PPO needs 4 models resident: policy, ref, reward, value):
    accelerate launch --config_file configs/accelerate/deepspeed_zero3_8gpu.yaml \\
        labs/03_ppo_rlhf/trl/train_ppo_trl.py
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.paths import REWARD_MODEL_NAME, SFT_MODEL_NAME, checkpoint_dir, run_dir  # noqa: E402
from common.utils import banner, set_seed  # noqa: E402

FALLBACK_POLICY_MODEL = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "trl-lib/tldr"
EVAL_SAMPLES = 200


def resolve_checkpoint(name: str, fallback: str, label: str) -> str:
    ckpt_dir = checkpoint_dir(name, create=False)
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        return str(ckpt_dir)
    print(f"[warn] no {label} checkpoint at {ckpt_dir}; falling back to {fallback}.")
    return fallback


def main() -> None:
    set_seed(0)
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    banner("Lab 03 (TRL): PPO on Qwen2.5-0.5B / TL;DR")

    policy_path = resolve_checkpoint(SFT_MODEL_NAME, FALLBACK_POLICY_MODEL, "lab-01 SFT")
    reward_path = resolve_checkpoint(REWARD_MODEL_NAME, None, "lab-02 reward model")
    if reward_path is None:
        raise SystemExit(
            "This script needs the lab-02 reward model. Run "
            "labs/02_reward_modeling/trl/train_rm_trl.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(policy_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    policy = AutoModelForCausalLM.from_pretrained(policy_path, dtype=torch.bfloat16)
    ref_policy = AutoModelForCausalLM.from_pretrained(policy_path, dtype=torch.bfloat16)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_path, num_labels=1, dtype=torch.bfloat16)
    value_model = AutoModelForSequenceClassification.from_pretrained(reward_path, num_labels=1, dtype=torch.bfloat16)

    dataset = load_dataset(DATASET_NAME, split="train")

    def tokenize(element):
        return {"input_ids": tokenizer(element["prompt"], padding=False)["input_ids"]}

    train_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names, num_proc=4)
    eval_dataset = (
        load_dataset(DATASET_NAME, split="validation")
        .select(range(EVAL_SAMPLES))
        .map(tokenize, batched=True, remove_columns=dataset.column_names, num_proc=4)
    )

    out_dir = checkpoint_dir("ppo_qwen2.5-0.5b_tldr")
    shutil.rmtree(out_dir, ignore_errors=True)
    args = PPOConfig(
        output_dir=str(out_dir),
        logging_dir=str(run_dir("lab03_ppo_trl")),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        total_episodes=10_000,
        num_ppo_epochs=4,
        num_mini_batches=2,
        response_length=64,
        kl_coef=0.05,  # same role as KL_BETA in the scratch script
        cliprange=0.2,
        vf_coef=0.1,
        missing_eos_penalty=1.0,  # penalize responses that never emit EOS (ramble past the limit)
        learning_rate=3e-6,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[r for r in ["wandb"] if os.environ.get("WANDB_PROJECT")] or ["tensorboard"],
        run_name="lab03_ppo_trl",
    )

    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    trainer.generate_completions()

    print(f"\nSaved PPO-tuned policy to {out_dir}")
    print("Compare its TL;DR summaries against the lab-01 SFT checkpoint and lab-04's DPO/KTO models.")


if __name__ == "__main__":
    main()
