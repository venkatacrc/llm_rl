# Lab 03 -- PPO / Classic RLHF

**Goal:** implement the InstructGPT recipe end-to-end -- rollout generation, reward
model scoring, a KL penalty against the reference policy, GAE, and the clipped PPO
surrogate loss -- and see it visibly shift a policy's behavior.

## Theory

RLHF-with-PPO treats text generation as a (very short-horizon, very high-branching)
RL problem: the "state" is the prompt + tokens generated so far, the "action" is the
next token, and the episode ends when generation stops. The pieces:

1. **Reward.** A trained reward model $r_\phi$ (lab 02) scores the *completed*
   response. This is a **sparse, terminal reward** -- there's no per-token ground truth,
   only a score for the whole completion.
2. **KL penalty.** Optimizing raw reward alone is dangerous: the policy can find
   degenerate text the reward model over-scores (reward hacking, lab 08). So the actual
   objective is:
   $$\max_\theta \; \mathbb{E}\big[r_\phi(x,y)\big] - \beta \, \mathbb{E}\big[\mathrm{KL}(\pi_\theta(\cdot|x) \,\|\, \pi_{ref}(\cdot|x))\big]$$
   Implemented as a **per-token** penalty `-beta * (log pi_theta(token) - log pi_ref(token))`
   added to the (otherwise zero) reward at every step, with the terminal RM score added
   only at the final token.
3. **GAE (Generalized Advantage Estimation).** A learned value head $V_\psi$ estimates
   expected future (KL-adjusted) reward from each prefix. GAE combines multi-step
   returns with the value baseline to get a lower-variance advantage estimate
   $\hat{A}_t$, trading off bias (small $\lambda$) vs. variance (large $\lambda$).
4. **Clipped surrogate loss.** Because a rollout batch is reused for several gradient
   steps (`PPO_EPOCHS`), the policy that generated the data (`old`) drifts from the
   policy being updated (`new`). PPO bounds how much any single update can exploit that
   drift:
   $$L^{CLIP}(\theta) = \mathbb{E}\Big[\min\big(\rho_t \hat{A}_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big], \quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$$

That's the whole algorithm. Everything else (value loss, entropy bonus, reward/advantage
normalization) is stabilization detail, present in `scratch/train_ppo_scratch.py` and
commented inline.

## Files

- `scratch/train_ppo_scratch.py` -- every piece above, hand-written, on
  `lvwerra/gpt2-imdb` (GPT-2 already SFT'd on movie reviews -- see lab 01 for how that
  step works) with `lvwerra/distilbert-imdb` as the reward model. Classic, fast-feedback
  demo: within ~100-150 PPO steps the model visibly shifts towards positive reviews.
- `trl/train_ppo_trl.py` -- the same algorithm via TRL's `PPOTrainer` (now under
  `trl.experimental.ppo`), on Qwen2.5-0.5B, using the reward model trained in lab 02 on
  the TL;DR summarization task.

## Run it

```bash
# from-scratch (single GPU, a few minutes)
python labs/03_ppo_rlhf/scratch/train_ppo_scratch.py

# TRL (run labs 01 and 02's TRL scripts first to get the SFT + reward-model checkpoints)
python labs/03_ppo_rlhf/trl/train_ppo_trl.py
```

Watch `mean_reward` climb and read the printed sample completion each log step --
you should see it drift from generic movie-review text towards enthusiastic, unambiguously
positive language.

## Things to try

- Set `KL_BETA = 0.0` in the scratch script and re-run: reward will climb *faster* at
  first, then the sample text will degrade into repetitive/nonsensical
  positive-sounding tokens -- textbook reward hacking (formalized in lab 08).
- Sweep `CLIP_EPS` (try 0.05 vs 0.4) and watch `clip_frac`/`approx_kl` in the logged
  metrics -- tighter clipping should keep `approx_kl` smaller step-to-step, at the cost
  of slower reward improvement.
- Compare `GAE_LAMBDA = 1.0` (pure Monte Carlo advantage, high variance) against
  `GAE_LAMBDA = 0.0` (pure TD/bootstrap, high bias) and see which trains more stably at
  this response length.
- After running both `scratch` and `trl` versions, sample outputs from all three
  models -- lab 01's SFT model, this lab's PPO model, and lab 04's DPO model -- on the
  same prompts to build intuition for how similarly (or not) PPO and DPO shift behavior.
