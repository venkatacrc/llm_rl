# Lab 02 -- Reward Modeling

**Goal:** understand how a scalar reward model is trained from *pairwise* human (or
rule-based) preferences via the Bradley-Terry model -- this is the reward signal PPO
(lab 03) optimizes, and the thing DPO (lab 04) shows you can skip entirely.

## Theory

We rarely have absolute quality scores for text ("this summary is a 7.3/10") -- what we
usually have is relative judgments: "summary A is better than summary B". The
Bradley-Terry model explains such comparisons as arising from a latent scalar score
(the reward) for each option:

$$P(A \succ B) = \sigma\big(r(A) - r(B)\big)$$

Given a dataset of `(chosen, rejected)` pairs, we fit a neural network $r_\phi$ (a
transformer with a scalar output head instead of a vocabulary head) by maximum
likelihood, i.e. minimizing the negative log-likelihood:

$$\mathcal{L}(\phi) = -\log \sigma\big(r_\phi(\text{chosen}) - r_\phi(\text{rejected})\big)$$

A few properties worth internalizing, since they explain a lot of "why does RLHF do
X" later on:

- **Only the *gap* matters.** The loss is invariant to adding a constant to every
  reward, so the RM's absolute scale is meaningless/arbitrary -- only relative
  comparisons and the trained scale (from how confidently pairs are separated) are
  meaningful. This is why reward normalization/centering shows up everywhere downstream.
- **The RM can be *very* wrong off-distribution.** It was only ever trained to compare
  pairs from roughly the SFT policy's distribution. Once PPO starts pushing the policy
  into new territory, the RM's judgments there are extrapolation, not learned
  preference -- this is the root cause of reward hacking (see lab 08).
- **This is exactly the mechanism behind DPO** (lab 04): substitute the Bradley-Terry
  reward with the implicit reward $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$
  and the same loss becomes a loss directly on the *policy* -- no separate reward model,
  no RL rollout loop needed. Understanding this lab's loss is 80% of understanding DPO.

## Files

- `scratch/train_rm_scratch.py` -- hand-written Bradley-Terry loss, GPT-2 with a linear
  scalar head (`AutoModelForSequenceClassification(num_labels=1)`), trained on preference
  pairs built from IMDB (`chosen` = a positive review, `rejected` = a negative one).
  Evaluated by pairwise accuracy on held-out pairs.
- `trl/train_rm_trl.py` -- the same objective via TRL's `RewardTrainer`, on Qwen2.5-0.5B
  initialized from lab 01's SFT checkpoint, trained on `trl-lib/tldr-preference`. Saves
  to `checkpoints/reward_model_qwen2.5-0.5b_tldr/`, used by lab 03 (PPO) and lab 08
  (reward hacking).

## Run it

```bash
# from-scratch (single GPU, a few minutes)
python labs/02_reward_modeling/scratch/train_rm_scratch.py

# TRL, on Qwen2.5-0.5B (run lab 01's TRL script first so this starts from the SFT model)
python labs/02_reward_modeling/trl/train_rm_trl.py
```

Watch the pairwise accuracy climb from ~0.5 (random) towards >0.85 -- that's the RM
learning to separate chosen from rejected purely from relative comparisons.

## Things to try

- In `train_rm_scratch.py`, print the actual reward scores for a few individual reviews
  (not just accuracy) -- notice they're not calibrated to any particular range; only
  their ordering matters.
- Swap `center_rewards_coefficient` in the TRL script to `0.0` and compare the
  distribution of predicted rewards before/after training (`trainer.predict`) -- you
  should see the mean drift away from 0 without the centering term.
- Try training the scratch RM with far fewer pairs (e.g. 200 instead of 4000) and watch
  the held-out pairwise accuracy degrade -- reward models are notoriously data-hungry
  compared to the policy they'll later train.
- Feed the trained scratch RM a review from a *different domain* (e.g. a restaurant
  review) and see how confidently (mis)calibrated its score is -- a hands-on preview of
  the out-of-distribution reward-hacking failure mode explored in lab 08.
