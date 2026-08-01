"""Lab 00, part 2: NCCL all-reduce smoke test + bandwidth check across all 8 GPUs.

Every multi-GPU lab (00's own throughput bench, lab 07's capstone) depends on NCCL
being able to talk to all 8 B200s over NVLink. This script launches one process per
GPU, does a correctness check (all-reduce a known tensor, verify the result), then
times all-reduce at a few tensor sizes to give you a rough NVLink bandwidth reading.

Run with torchrun so one process is spawned per GPU:

    torchrun --standalone --nproc_per_node=8 labs/00_setup_and_gpu_tour/nccl_smoke_test.py

Expect NVLink (not PCIe) bandwidth: multiple hundreds of GB/s for large tensors on a
healthy B200 node.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)

    # --- Correctness check ---
    x = torch.full((4,), fill_value=float(rank + 1), device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = sum(r + 1 for r in range(world_size))
    ok = torch.allclose(x, torch.full_like(x, expected))
    if rank == 0:
        print(f"[correctness] all_reduce(sum of rank+1 across {world_size} ranks) == {expected}: {ok}")
    assert ok, "NCCL all-reduce gave the wrong answer -- something is badly wrong with the setup."

    dist.barrier()

    # --- Bandwidth check across a few sizes ---
    if rank == 0:
        print(f"\n{'tensor size':>14s} | {'time/iter (ms)':>15s} | {'algbw (GB/s)':>13s}")
    for numel in [1_000_000, 10_000_000, 100_000_000, 500_000_000]:
        t = torch.randn(numel, device=device, dtype=torch.bfloat16)
        n_iters = 10
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = (time.perf_counter() - t0) / n_iters
        bytes_moved = numel * 2  # bf16 = 2 bytes/elem; all-reduce ~= 2x(N-1)/N * bytes, we report the simple algbw
        algbw_gbps = bytes_moved / elapsed / 1e9
        if rank == 0:
            print(f"{numel:>14,d} | {elapsed * 1000:>15.3f} | {algbw_gbps:>13.1f}")
        del t
        torch.cuda.empty_cache()

    dist.destroy_process_group()
    if rank == 0:
        print("\nNCCL smoke test passed on all ranks.")


if __name__ == "__main__":
    main()
