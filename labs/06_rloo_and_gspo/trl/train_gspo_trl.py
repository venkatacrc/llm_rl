"""Lab 06 (at scale): GSPO -- sequence-level importance sampling for GRPO.

GSPO (Group Sequence Policy Optimization, from Qwen) makes one change to GRPO: instead
of computing the PPO-style importance ratio `pi_theta(token)/pi_old(token)`
*per token*, it averages the log-ratio across the whole completion to get a single
*per-sequence* ratio:

    token-level (GRPO):  rho_t = exp(logpi_theta(y_t) - logpi_old(y_t))                for each t
    sequence-level (GSPO): rho = exp( (1/|y|) * sum_t [logpi_theta(y_t) - logpi_old(y_t)] )

The motivation: GRPO's advantage is a single *sequence-level* number broadcast over all
tokens, but its importance-sampling correction is applied per token -- a mismatch the
GSPO paper argues causes training instability, especially for very long completions or
Mixture-of-Experts models (where token-level routing adds extra noise to per-token
ratios). Matching the importance-sampling granularity to the granularity of the
advantage (both sequence-level) is the fix.

In TRL, this is a `GRPOConfig` flag, not a separate trainer:

    GRPOConfig(importance_sampling_level="sequence", loss_type="grpo", ...)

(the `loss_type="grpo"` -- rather than the default `"dapo"` -- matters here: token-summed
loss types combined with sequence-level importance sampling silently length-weight each
sequence, which is *not* what the GSPO paper does; see TRL's paper index.) Everything
else is identical to lab 05's GRPO run, so run both and compare `runs/lab05_grpo_trl`
vs `runs/lab06_gspo_trl`.

Single GPU:
    python labs/06_rloo_and_gspo/trl/train_gspo_trl.py
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
    from trl import GRPOConfig, GRPOTrainer

    banner(f"Lab 06 (TRL): GSPO (sequence-level IS) on {MODEL_NAME} / GSM8K")

    train_dataset = load_gsm8k("train")
    eval_dataset = load_gsm8k("test").select(range(200))

    out_dir = checkpoint_dir("gspo_qwen2.5-math-1.5b_gsm8k")
    args = GRPOConfig(
        output_dir=str(out_dir),
        logging_dir=str(run_dir("lab06_gspo_trl")),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        steps_per_generation=4,  # must be a multiple of gradient_accumulation_steps; off-policy-ness is
        num_generations=8,  # what makes sequence- vs token-level importance sampling actually differ
        max_prompt_length=256,
        max_completion_length=512,
        learning_rate=1e-6,
        # --- The GSPO recipe (see TRL's paper index for exact provenance of these values) ---
        importance_sampling_level="sequence",
        loss_type="grpo",
        beta=0.0,  # GSPO paper sets the KL coefficient to 0
        epsilon=3e-4,
        epsilon_high=4e-4,
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
        run_name="lab06_gspo_trl",
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

    print(f"\nSaved GSPO-tuned model to {out_dir}")
    print(
        "Compare runs/lab06_gspo_trl against runs/lab05_grpo_trl (token-level GRPO) -- with "
        "steps_per_generation > gradient_accumulation_steps, training is mildly off-policy, which "
        "is precisely the regime where sequence- vs token-level importance sampling can diverge."
    )


if __name__ == "__main__":
    main()
