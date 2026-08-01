# Lab 00 -- Setup and GPU Tour

**Goal:** confirm the whole stack (torch, CUDA 13.0, all 8 B200s, transformers, trl,
vLLM) works before you spend any time debugging an RL algorithm that's actually failing
because of an environment issue.

No RL yet -- this lab is pure infrastructure. But it's worth doing properly: three of
the most common "my RLHF run is behaving weirdly" bugs are actually (a) NCCL/interconnect
issues, (b) an accidentally-CPU or accidentally-fp32 model, and (c) a mismatched
torch/vllm CUDA build. This lab catches all three up front.

## 1. Environment check

```bash
python labs/00_setup_and_gpu_tour/check_env.py
```

Checks python/torch/CUDA versions, that all 8 GPUs are visible with ~183GB each, bf16
support, and that transformers/accelerate/trl/peft/datasets/deepspeed/vllm all import
with reasonably current versions. Prints PASS/WARN/FAIL and exits non-zero on any FAIL.

## 2. NCCL smoke test (needs all 8 GPUs)

```bash
torchrun --standalone --nproc_per_node=8 labs/00_setup_and_gpu_tour/nccl_smoke_test.py
```

Spawns one process per GPU, does a correctness-checked all-reduce, then times
all-reduce at a few tensor sizes as a rough NVLink bandwidth reading. On a healthy
B200 node you should see bandwidth in the hundreds of GB/s for the larger tensors; if
you instead see PCIe-like numbers (tens of GB/s), something's wrong with NVLink/topology
and it's worth running `nvidia-smi topo -m` to check.

## 3. Throughput benchmark

```bash
# single GPU baseline
python labs/00_setup_and_gpu_tour/throughput_benchmark.py

# 8-way data-parallel (training throughput should scale close to linearly)
torchrun --standalone --nproc_per_node=8 labs/00_setup_and_gpu_tour/throughput_benchmark.py
```

Measures (a) forward+backward tokens/sec training Qwen2.5-0.5B, and (b) tokens/sec
generating with plain `transformers.generate`. Keep the single-GPU numbers handy --
lab 07's scaling benchmark and the vLLM-vs-`.generate()` comparison in lab 03+ are both
more meaningful with a baseline to compare against.

## 4. Hello, generate

```bash
python labs/00_setup_and_gpu_tour/hello_generate.py
```

Loads Qwen2.5-0.5B-Instruct, applies the chat template, and generates a few responses.
If this produces three coherent answers, chat-template handling and generation both
work end-to-end.

## Things to try

- Run `nvidia-smi dmon -s u` in another terminal while the throughput benchmark runs,
  to watch utilization/power directly.
- Compare `check_env.py`'s reported `torch.version.cuda` against your driver's
  `nvidia-smi` CUDA version (13.0) -- torch wheels are typically built for an older CUDA
  toolkit than your driver supports, and that's fine (forward compatible), but it's good
  to know what you're running.
- Re-run the throughput benchmark with `attn_implementation="eager"` vs `"sdpa"` (edit
  the script) to see the effect of the fused attention kernel.
