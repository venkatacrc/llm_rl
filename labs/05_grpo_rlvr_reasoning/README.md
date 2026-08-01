# Lab 05 -- GRPO and RLVR Reasoning

**Goal:** understand the two big ideas behind how models like DeepSeek-R1 are trained:
(1) **GRPO** -- replacing PPO's learned critic with a cheap, group-relative baseline,
and (2) **RLVR** (RL from Verifiable Rewards) -- replacing the learned reward model
with a rule that can check correctness directly (does the final answer match?).

## Theory

**Why drop the critic?** PPO's value function $V_\psi(s)$ exists to provide a
low-variance baseline for the advantage estimate. Training it well is itself hard
(it's a whole extra network, prone to being wrong especially early in training) and it
roughly doubles memory/compute (a second copy of the model, effectively). GRPO's
insight: if you sample **multiple completions for the same prompt**, you get a
*for-free*, unbiased baseline -- the group's own mean reward:

$$A_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G) + \epsilon}, \qquad i = 1, \dots, G$$

No value network, no GAE -- just: sample a group, score each member, normalize within
the group. This only requires the reward to be reasonably cheap to compute many times
per prompt, which brings us to:

**Why verifiable rewards?** A learned reward model (lab 02) is itself a noisy,
exploitable proxy (lab 08). For tasks with a checkable ground truth -- math (does the
number match?), code (do the tests pass?) -- you can skip the reward model entirely and
score with a rule. This is dramatically cheaper (no extra forward pass through a big
network per sample) and, crucially, **can't be reward-hacked** in the same way a
learned model can, since it's checking the actual thing you care about.

**The KL term moves.** PPO subtracts a per-token KL penalty from the reward
before computing advantages (lab 03). GRPO instead adds a KL term directly to the
*loss*, using Schulman's k3 estimator (unbiased, always $\geq 0$, low variance):

$$\widehat{\mathrm{KL}}_{k3}(\pi_\theta \| \pi_{ref}) = \exp(\log\pi_{ref} - \log\pi_\theta) - (\log\pi_{ref} - \log\pi_\theta) - 1$$

Otherwise the update is the same clipped-surrogate machinery as PPO.

## Files

- `env/math_reward.py` -- GSM8K loading + a rule-based reward: extract the model's
  final numeric answer, compare to ground truth (reused from `common/rewards.py`).
- `scratch/train_grpo_scratch.py` -- GRPO hand-written on GPT-2, task: single-digit
  addition (`"3 + 5 = "` -> `8`). Deliberately tiny/fast so you can watch group
  statistics (mean reward, fraction of "zero-signal" groups) evolve over a few hundred
  steps on one GPU.
- `trl/train_grpo_trl.py` -- `GRPOTrainer` + vLLM (colocate mode) on
  `Qwen/Qwen2.5-Math-1.5B` with real GSM8K math word problems -- the DeepSeek-R1-Zero
  recipe at a scale you can actually run.

## Run it

```bash
# from-scratch (single GPU, a few minutes)
python labs/05_grpo_rlvr_reasoning/scratch/train_grpo_scratch.py

# TRL + vLLM on Qwen2.5-Math-1.5B (single GPU)
python labs/05_grpo_rlvr_reasoning/trl/train_grpo_trl.py

# multi-GPU (more groups in flight per step)
accelerate launch --config_file configs/accelerate/deepspeed_zero3_8gpu.yaml \
    labs/05_grpo_rlvr_reasoning/trl/train_grpo_trl.py
```

## Things to try

- Watch `zero_signal_group_frac` in the scratch run's metrics: early on, most groups
  are "all wrong" (0 variance, 0 gradient); track how it shrinks as accuracy improves,
  then note it *rises again* as accuracy approaches ~100% (now most groups are "all
  right"). GRPO's signal is richest in the middle of the accuracy curve -- an important,
  underappreciated property of group-relative RLVR.
- Remove the `/ (group_std + eps)` normalization (Dr. GRPO's fix for a length/scale
  bias this introduces) and compare training stability -- see `paper_index` in TRL's
  docs for the precise argument.
- Increase `GROUP_SIZE` (e.g. 4 -> 16) in the scratch script and see whether fewer
  total steps are needed to reach the same accuracy (more samples per prompt = better
  baseline estimate, at the cost of more compute per step).
- In the TRL script, try removing the `#### <number>` format bonus from
  `env/math_reward.py`'s `gsm8k_reward` and see whether/how much slower the model
  converges to a parseable answer format.
- Compare wall-clock time for the TRL script with `use_vllm=False` vs. `True` -- this
  is the single biggest lever for RLVR training speed, since rollout generation
  (not the gradient step) dominates wall-clock time once group sizes get large.
