# Lab 04 -- The DPO Family

**Goal:** derive Direct Preference Optimization from the same Bradley-Terry model lab
02 used for reward modeling, see that it eliminates the reward model *and* the RL
rollout loop entirely, then compare it (and its cousin KTO) against lab 03's PPO on a
shared reward-vs-KL frontier.

## Theory

Lab 02's reward-model loss:

$$\mathcal{L}_{RM}(\phi) = -\log\sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$$

DPO's derivation starts from a fact about the KL-constrained RL objective PPO
(lab 03) optimizes -- $\max_\theta \mathbb{E}[r(x,y)] - \beta \, \mathrm{KL}(\pi_\theta \| \pi_{ref})$
-- which has a *closed-form optimal policy*:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\!\big(r(x,y)/\beta\big)$$

Solving this for $r$ gives $r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)$
-- i.e. *any* reward function is equivalent (up to a prompt-dependent constant that
cancels in a pairwise comparison) to the log-ratio of some policy over the reference.
Substituting this "implicit reward" into the Bradley-Terry loss above and dropping the
now-cancelled $Z(x)$ terms gives the **DPO loss**, directly over policy parameters:

$$\mathcal{L}_{DPO}(\theta) = -\log\sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\big]\Big)$$

No reward model, no sampling from $\pi_\theta$ during training, no value function --
just two forward passes (policy and frozen reference) over already-collected
`(prompt, chosen, rejected)` triples, and ordinary supervised-style backprop. The
trade-off: DPO is fully **off-policy** (it only ever "sees" the fixed dataset's
completions, never the current policy's own samples), whereas PPO/GRPO are (at least
partially) **on-policy**, which matters when the offline preference data doesn't cover
the region the policy needs to move into.

**KTO** relaxes DPO's need for *paired* preferences: instead of "y_w beats y_l for
this x", it only needs independent binary desirable/undesirable labels per
`(prompt, completion)`, using a prospect-theory-inspired asymmetric loss around a
reference point. Practically: pass the same paired data through unchanged and
`KTOTrainer` will split it into unpaired examples for you (see `trl/train_kto_trl.py`).

## Files

- `scratch/train_dpo_scratch.py` -- the DPO loss above, hand-coded, on
  `lvwerra/gpt2-imdb` with the same IMDB-derived preference pairs style as lab 02's
  scratch reward model (so it's directly comparable to lab 03's PPO run, which starts
  from the same base model).
- `trl/train_dpo_trl.py` / `trl/train_kto_trl.py` -- `DPOTrainer` / `KTOTrainer` on
  Qwen2.5-0.5B (from the lab-01 SFT checkpoint), on the exact same
  `trl-lib/tldr-preference` data lab 02 used for reward modeling.
- `trl/compare_ppo_dpo_kto.py` -- generates from whichever of {SFT, PPO, DPO, KTO}
  checkpoints exist, scores them with lab 02's reward model, estimates each one's KL
  drift from the SFT reference, and plots the reward-vs-KL frontier.

## Run it

```bash
# from-scratch (single GPU, a few minutes)
python labs/04_dpo_family/scratch/train_dpo_scratch.py

# TRL (run lab 01's TRL script first for the SFT starting checkpoint)
python labs/04_dpo_family/trl/train_dpo_trl.py
python labs/04_dpo_family/trl/train_kto_trl.py

# after running lab 03's TRL PPO script too, compare all three:
python labs/04_dpo_family/trl/compare_ppo_dpo_kto.py
```

## Things to try

- In the scratch script, set `BETA = 0.01` vs `BETA = 1.0` and compare the final
  `margin` (chosen_reward - rejected_reward) and held-out accuracy -- beta controls how
  sharply DPO is willing to move the policy per unit of preference signal, exactly like
  the KL coefficient in PPO.
- Derive on paper (or in a scratch cell) what happens to the DPO loss as
  $\pi_\theta \to \pi_{ref}$ everywhere (hint: the gradient vanishes) -- this is why DPO
  still needs *some* initial preference signal exploitable near the reference policy.
- Run `compare_ppo_dpo_kto.py` after only training DPO (skip PPO/KTO) and re-run again
  once you've trained the others -- the script gracefully skips missing checkpoints, so
  you can build up the comparison incrementally.
- Try training DPO directly from the raw `Qwen/Qwen2.5-0.5B` base model (skip lab 01's
  SFT step) and compare sample quality/coherence against starting from the SFT
  checkpoint -- this is a hands-on version of the "why SFT first" argument from lab 01.
