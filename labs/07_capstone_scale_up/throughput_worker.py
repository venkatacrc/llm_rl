"""Lab 07 helper: single training-throughput measurement, launched via torchrun.

Not meant to be run directly -- `benchmark_scaling.py` launches this once per GPU count
via `torchrun --standalone --nproc_per_node=<N>` and parses the
`THROUGHPUT_TOKENS_PER_SEC=<value>` line it prints on rank 0.

Uses DDP (not FSDP) deliberately: a 7B model in bf16 with AdamW (weights + grads + 2
optimizer moments, roughly 16 bytes/param all-in) is ~110GB -- comfortably within a
single B200's 183GB, so full replication is actually fine here and keeps the benchmark
apples-to-apples across 1/2/4/8 GPUs (FSDP would change the per-GPU memory/compute
profile as the shard count changes, confounding the scaling measurement itself).
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.utils import set_seed  # noqa: E402

MODEL_NAME = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SEQ_LEN = 512
BATCH_SIZE_PER_GPU = 2
N_WARMUP = 3
N_ITERS = 10


def main() -> None:
    set_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if distributed:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.gradient_checkpointing_enable()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    input_ids = torch.randint(0, tokenizer.vocab_size, (BATCH_SIZE_PER_GPU, SEQ_LEN), device=device)
    labels = input_ids.clone()

    def step():
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optim.step()
        optim.zero_grad()

    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()
    if distributed:
        import torch.distributed as dist

        dist.barrier()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        step()
    torch.cuda.synchronize()
    if distributed:
        import torch.distributed as dist

        dist.barrier()
    elapsed = time.perf_counter() - t0

    local_tokens_per_sec = (BATCH_SIZE_PER_GPU * SEQ_LEN * N_ITERS) / elapsed
    if distributed:
        import torch.distributed as dist

        t = torch.tensor([local_tokens_per_sec], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_tokens_per_sec = t.item()
        dist.destroy_process_group()
    else:
        total_tokens_per_sec = local_tokens_per_sec

    if local_rank == 0:
        print(f"THROUGHPUT_TOKENS_PER_SEC={total_tokens_per_sec:.1f}")


if __name__ == "__main__":
    main()
