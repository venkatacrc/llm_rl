# Lab 06 -- RLOO and GSPO Variants

**Goal:** see two more points in the PPO/GRPO design space: RLOO (how much of PPO's
machinery can you strip away and still learn?) and GSPO (matching the *granularity* of
importance sampling to the granularity of the advantage).

## Theory

**RLOO** asks: do we need PPO's clipped surrogate and multi-epoch reuse at all? Its
answer is closer to the original REINFORCE algorithm:

$$\nabla_\theta J(\theta) = \mathbb{E}\big[(R - b)\, \nabla_\theta \log \pi_\theta(y|x)\big]$$

with the baseline $b$ chosen as the **leave-one-out** mean of a sampled group:

$$b_i = \frac{1}{G-1}\sum_{j \neq i} r_j, \qquad A_i = r_i - b_i$$

Excluding $r_i$ from its own baseline is what makes this *exactly* unbiased (the
baseline is independent of the term it's subtracted from). GRPO's group-*mean* baseline
$r_i - \mathrm{mean}(r_1..r_G)$ includes $r_i$ in its own baseline, which introduces a
small $O(1/G)$ bias -- negligible for reasonable group sizes, but a genuine, precise
distinction. The much bigger practical difference in this lab's scratch script:
**variance**. Compare the logged `var_naive` (no baseline) against `var_loo` /
`var_group_mean` and you'll see any reasonable baseline is a large win over none at all.

**GSPO** asks a different question: PPO/GRPO's clipped objective needs an importance
ratio $\rho = \pi_\theta(\cdot)/\pi_{old}(\cdot)$ to correct for the policy having moved
since the rollout was sampled. GRPO computes this *per token*, but the advantage it's
multiplied by is a single *per-sequence* number (from the group). GSPO's fix: compute
the ratio at the *sequence* level too (average the log-ratio over the completion, then
exponentiate), so both halves of the update operate at the same granularity:

$$\rho^{\text{GSPO}} = \exp\!\Big(\tfrac{1}{|y|}\sum_{t} \log\tfrac{\pi_\theta(y_t)}{\pi_{old}(y_t)}\Big) \quad \text{vs.} \quad \rho^{\text{GRPO}}_t = \tfrac{\pi_\theta(y_t)}{\pi_{old}(y_t)} \; \text{(per token)}$$

This matters most when training is even mildly off-policy (rollouts reused for more
than one gradient step) and for very long completions or MoE models, where per-token
ratios get noisy.

## Files

- `scratch/train_rloo_scratch.py` -- vanilla REINFORCE + leave-one-out baseline, on the
  same single-digit-addition RLVR task as lab 05, logging naive/group-mean/LOO advantage
  variance side-by-side every step.
- `trl/train_rloo_trl.py` -- `RLOOTrainer` on the same GSM8K/Qwen2.5-Math-1.5B setup as
  lab 05's GRPO run, for a direct training-curve comparison.
- `trl/train_gspo_trl.py` -- `GRPOTrainer` with `importance_sampling_level="sequence"`,
  `loss_type="grpo"` (the TRL-documented GSPO recipe), same task/model as lab 05.

## Run it

```bash
# from-scratch (single GPU, a few minutes)
python labs/06_rloo_and_gspo/scratch/train_rloo_scratch.py

# TRL, on Qwen2.5-Math-1.5B / GSM8K
python labs/06_rloo_and_gspo/trl/train_rloo_trl.py
python labs/06_rloo_and_gspo/trl/train_gspo_trl.py
```

## Things to try

- Overlay `runs/lab05_grpo_scratch/metrics.csv` and `runs/lab06_rloo_scratch/metrics.csv`
  (`mean_reward` vs step) -- on this simple task, do GRPO and RLOO converge at a
  meaningfully different rate?
- In the scratch script, print the *actual* numeric difference between
  `adv_group_mean` and `adv_loo` for a few groups -- confirm it shrinks as `GROUP_SIZE`
  grows (the $O(1/G)$ bias argument, made concrete).
- In `train_gspo_trl.py`, set `steps_per_generation = gradient_accumulation_steps` (i.e.
  fully on-policy) and note TRL's own docs point out sequence- vs token-level importance
  sampling become *equivalent* in that regime -- the difference only shows up once
  training goes even mildly off-policy.
- Try `loss_type="bnpo"` instead of `"grpo"` in the GSPO script and watch for TRL's
  logged warning about token-summed loss types silently length-weighting sequences when
  combined with sequence-level importance sampling.
