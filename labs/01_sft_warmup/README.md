# Lab 01 -- SFT Warmup

**Goal:** understand why every RLHF/RLVR pipeline starts with supervised fine-tuning
(SFT), and build/run the training loop (prompt masking, loss computation) that every
later lab either reuses directly or conceptually depends on.

## Theory

Post-training pipelines (InstructGPT, DeepSeek-R1, Llama-3, ...) almost universally go:

```
pretrained base model --(SFT)--> reference/initial policy π_ref --(RL)--> aligned policy π_θ
```

Why not skip straight to RL from the raw pretrained model? Two reasons:

1. **Sample efficiency.** RL only ever gets a scalar reward per rollout -- it has to
   discover good behavior through trial and error. Starting from a policy that already
   roughly knows the target format/behavior (e.g. "answer the question", "write a
   summary") means RL only has to *shift a distribution*, not *invent one from scratch*.
2. **KL regularization needs a good reference.** Every RL algorithm in this curriculum
   (PPO, GRPO, DPO, RLOO...) penalizes the policy for drifting too far (in KL
   divergence) from a reference policy π_ref, both to avoid degenerate reward-hacking
   text and to keep the model's other capabilities intact. If π_ref is a raw
   pretrained model, that reference is a bad anchor.

**The mechanic that matters:** given a `(prompt, completion)` pair, you tokenize
`prompt + completion`, but only compute the next-token cross-entropy loss on the
*completion* tokens (mask the prompt to `-100`). If you don't mask the prompt, the model
also gets gradient signal for "predicting the prompt", which is not what you want (you
don't control the prompt at inference time) and wastes capacity/skews the loss scale.

## Files

- `scratch/train_sft_scratch.py` -- hand-written training loop: tokenize, mask, compute
  loss, AdamW step, periodically generate to eyeball progress. Model: GPT-2 (124M).
  Task: Reddit TL;DR summarization (`trl-lib/tldr`).
- `trl/train_sft_trl.py` -- the same idea via TRL's `SFTTrainer`, on Qwen2.5-0.5B. Saves
  its output to `checkpoints/sft_qwen2.5-0.5b_tldr/`, which labs 02-06's TRL scripts load
  as their starting policy.

## Run it

```bash
# from-scratch (single GPU, ~few minutes)
python labs/01_sft_warmup/scratch/train_sft_scratch.py

# TRL, on Qwen2.5-0.5B (single GPU)
python labs/01_sft_warmup/trl/train_sft_trl.py

# TRL, data-parallel across all 8 GPUs
accelerate launch --config_file configs/accelerate/ddp_8gpu.yaml \
    labs/01_sft_warmup/trl/train_sft_trl.py
```

Both scripts print sample summaries before/during/after training so you can watch the
model's outputs get shorter and more on-topic. Metrics land in
`runs/lab01_sft_scratch/metrics.csv` (+ a PNG) and `runs/lab01_sft_trl/` (TensorBoard),
respectively.

## Things to try

- In `train_sft_scratch.py`, comment out the prompt-masking (set `labels = input_ids`
  entirely) and compare the loss curve and generated samples -- you should see the loss
  drop faster (predicting the prompt is easy, it's in the input) while the actual
  completions get *worse*, a good illustration of why masking matters.
- Increase `N_TRAIN_EXAMPLES` / `N_EPOCHS` in the scratch script and see how much more
  coherent the summaries get.
- In the TRL script, try `Qwen/Qwen2.5-1.5B` instead of the 0.5B model (just change
  `MODEL_NAME`) and compare sample quality at the same training budget.
- Swap in `HuggingFaceH4/ultrachat_200k` (a conversational dataset) instead of
  `trl-lib/tldr` in the TRL script to see general instruction-following SFT instead of
  a single narrow task -- note `SFTTrainer` auto-applies the chat template for
  conversational-format datasets.
