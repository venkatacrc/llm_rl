# Lab 07 -- Capstone: Scale-Up

**Goal:** take the GRPO/RLVR recipe from lab 05 and run it "for real" -- a 7-8B model,
sharded across all 8 B200s, with vLLM colocated for rollout generation -- and measure
exactly what scaling the node buys you.

This lab is intentionally light on *new* algorithmic content: the point is that scaling
RL post-training from a 1.5B toy run to a real model is mostly an **infrastructure**
problem (sharding, mixed precision, colocated generation, checkpointing) layered on top
of the *exact same* algorithm you already understand from lab 05.

## Files

- `train_grpo_capstone.py` -- lab 05's GRPO + GSM8K recipe, parameterized for a larger
  model (`Qwen/Qwen2.5-7B-Instruct` by default), meant to be launched with `accelerate`
  across multiple GPUs.
- `throughput_worker.py` / `benchmark_scaling.py` -- a controlled 1/2/4/8-GPU training
  throughput benchmark (data-parallel, not FSDP, so the per-GPU compute/memory profile
  stays constant across GPU counts and the comparison is apples-to-apples).

## Run it

```bash
# the capstone training run itself, sharded via FSDP across all 8 GPUs
accelerate launch --config_file configs/accelerate/fsdp_8gpu.yaml \
    labs/07_capstone_scale_up/train_grpo_capstone.py

# if FSDP + colocated vLLM misbehaves on your installed versions, DeepSpeed ZeRO-3
# is the alternative (same as labs 05/06 already use):
accelerate launch --config_file configs/accelerate/deepspeed_zero3_8gpu.yaml \
    labs/07_capstone_scale_up/train_grpo_capstone.py

# throughput scaling benchmark (needs all 8 GPUs free, ~a few minutes)
python labs/07_capstone_scale_up/benchmark_scaling.py
```

`train_grpo_capstone.py` accepts `--model`, `--per-device-batch-size`,
`--num-generations`, `--gradient-accumulation-steps`, and
`--vllm-gpu-memory-utilization` if you want to point it at a different model size or
tune memory usage -- see lab 00's throughput benchmark and the OOM-debugging notes in
lab 05's README for guidance on what to adjust first.

## Things to try

- Run `benchmark_scaling.py --model Qwen/Qwen2.5-0.5B-Instruct` (matches lab 00's
  model) and compare scaling efficiency against the default 7B model -- smaller models
  are more likely to be communication-bound (gradient all-reduce dominates), so you may
  see *worse* scaling efficiency at 8 GPUs for the small model than the large one.
- Watch `nvidia-smi dmon` during the capstone run: with vLLM colocated, you should see
  GPU utilization alternate between "training" and "generation" phases rather than a
  flat 100% -- that's the colocate/sleep-mode handoff happening.
- Try `--vllm-gpu-memory-utilization 0.2` vs `0.6` and see the effect on both OOM risk
  and rollout throughput (more vLLM memory = larger KV cache = more concurrent
  sequences = faster generation, at the cost of less headroom for the training step).
- Swap `Qwen/Qwen2.5-7B-Instruct` for `meta-llama/Llama-3.1-8B-Instruct` (you'll need to
  accept its license on the Hugging Face Hub first) and compare GSM8K learning curves
  between the two base models.
