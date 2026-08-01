"""Lab 00, part 3: single-GPU vs multi-GPU throughput micro-benchmark.

Two things are measured, both relevant to every later lab:
  1. Training throughput (forward+backward, tokens/sec) on Qwen2.5-0.5B, with and
     without DDP across multiple GPUs.
  2. Generation throughput (tokens/sec) with plain `transformers.generate`, which is
     the "slow path" the `trl/` scripts replace with vLLM starting in lab 03.

Single GPU:
    python labs/00_setup_and_gpu_tour/throughput_benchmark.py

Data-parallel across all 8 GPUs (scales training throughput ~linearly; generation
throughput here is still measured per-process, so multiply by world size):
    torchrun --standalone --nproc_per_node=8 labs/00_setup_and_gpu_tour/throughput_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.utils import banner, set_seed  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def is_distributed() -> bool:
    return "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def training_throughput(model, device, vocab_size: int, seq_len: int = 512, batch_size: int = 8, n_iters: int = 20) -> float:
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    for _ in range(3):  # warmup
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optim.step()
        optim.zero_grad()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optim.step()
        optim.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = batch_size * seq_len * n_iters
    return tokens / elapsed


@torch.no_grad()
def generation_throughput(model, tokenizer, device, prompt_len: int = 64, new_tokens: int = 128, batch_size: int = 8, n_iters: int = 3) -> float:
    model.eval()
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, prompt_len), device=device)
    attn_mask = torch.ones_like(input_ids)

    model.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=8, do_sample=False)  # warmup
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = batch_size * new_tokens * n_iters
    return tokens / elapsed


def main() -> None:
    set_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    distributed = is_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    device = torch.device("cuda", local_rank)
    if rank == 0:
        banner(f"Throughput benchmark: {MODEL_NAME} on {world_size} GPU(s)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    train_tps = training_throughput(model, device, vocab_size=tokenizer.vocab_size)
    gen_tps = generation_throughput(model.module if distributed else model, tokenizer, device)

    if distributed:
        import torch.distributed as dist

        agg = torch.tensor([train_tps, gen_tps], device=device)
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        train_tps_total, gen_tps_total = agg.tolist()
    else:
        train_tps_total, gen_tps_total = train_tps, gen_tps

    if rank == 0:
        print(f"\nPer-GPU training throughput  : {train_tps:,.0f} tokens/sec")
        print(f"Per-GPU generation throughput: {gen_tps:,.0f} tokens/sec (greedy, transformers .generate)")
        if world_size > 1:
            print(f"\nAggregate (x{world_size} GPUs) training throughput  : {train_tps_total:,.0f} tokens/sec")
            print(f"Aggregate (x{world_size} GPUs) generation throughput: {gen_tps_total:,.0f} tokens/sec")
        print(
            "\nNote: generation here uses plain HF `.generate` (no KV-cache batching tricks, no "
            "continuous batching). Lab 03+ switch rollout generation to vLLM, which is typically "
            "5-20x faster -- that comparison is worth re-running once you get there."
        )

    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
