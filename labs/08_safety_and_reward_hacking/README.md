# Lab 08 -- Safety and Reward Hacking

**Goal:** see the failure modes every previous lab has been warning about, concretely:
reward hacking (Goodhart's law), the reward-vs-KL trade-off every algorithm makes
implicitly, the "alignment tax" on general capability, and a minimal red-teaming
harness for measuring whether alignment actually changed model behavior.

## Theory recap

Every lab so far mentioned a version of the same warning:

- Lab 02: the reward model is only trained on pairs from roughly the SFT policy's
  distribution -- once the policy moves elsewhere, its judgments are extrapolation.
- Lab 03: the KL penalty exists specifically to stop the policy from drifting into
  regions where the reward model is wrong-but-high.
- Lab 05: verifiable rewards are *harder* to hack (they check the real thing), but not
  impossible (a model can learn to satisfy the letter of a format check without the
  substance).

This lab makes each of these failure modes something you actually watch happen.

## Files

- `reward_hacking_demo.py` -- trains against a deliberately naive proxy reward
  (completion length) with little/no KL constraint, while separately monitoring (never
  training against) a sentiment-classifier score and a text-repetition metric. Watch the
  proxy reward climb while the monitored "true quality" signals stagnate or the text
  visibly degrades into repetitive filler.
- `kl_reward_frontier.py` -- trains several short runs at different KL coefficients
  (beta) against the *real* sentiment reward, and plots the resulting
  reward-vs-KL Pareto frontier.
- `alignment_tax.py` -- scores every available checkpoint (base, SFT, PPO, DPO, KTO)
  on a held-out MMLU subset via log-likelihood multiple-choice scoring, to see whether
  task-specific alignment cost anything on unrelated general knowledge.
- `redteam_eval.py` -- a fixed set of adversarial prompts + an LLM judge classifying
  each response as REFUSED/COMPLIED, tallying a refusal rate per checkpoint.

## Run it

```bash
python labs/08_safety_and_reward_hacking/reward_hacking_demo.py                  # kl_beta=0.0: watch it hack
python labs/08_safety_and_reward_hacking/reward_hacking_demo.py --kl-beta 0.2    # constrained: compare

python labs/08_safety_and_reward_hacking/kl_reward_frontier.py

# after training whichever of labs 01/03/04 you want compared:
python labs/08_safety_and_reward_hacking/alignment_tax.py
python labs/08_safety_and_reward_hacking/redteam_eval.py
```

## Things to try

- In `reward_hacking_demo.py`, plot `proxy_reward` and `true_reward_sentiment` on the
  same axes (they're both in `metrics.csv`) -- the growing *gap* between them, not
  either curve alone, is the real signature of reward hacking.
- Re-run `reward_hacking_demo.py` with a few different `--kl-beta` values (0.0, 0.05,
  0.2, 1.0) and see how much KL constraint is "enough" to prevent the length-hack on
  this task -- compare against where that beta value would land on
  `kl_reward_frontier.py`'s curve.
- In `alignment_tax.py`, swap in a domain-relevant eval instead of MMLU (e.g. a
  summarization-quality metric like ROUGE against reference TL;DRs) -- MMLU is a poor
  match for a model that was only ever trained/aligned on summarization, so a
  domain-specific eval will show the tax (or gain!) more clearly than general knowledge.
- In `redteam_eval.py`, read a handful of full transcripts by hand (not just the judge's
  verdict) -- LLM judges have their own failure modes (e.g. being fooled by a
  confident-sounding refusal that actually leaks some information), and it's worth
  building the habit of spot-checking automated evals.
- Combine ideas: run `redteam_eval.py` against a model you deliberately trained with
  `reward_hacking_demo.py`'s beta=0.0 setting (reward-hacked, degenerate text) and see
  how the judge handles clearly broken/repetitive outputs -- a reminder that automated
  evals can behave unpredictably outside their expected input distribution too.
