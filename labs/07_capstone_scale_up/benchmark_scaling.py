"""Lab 07: 1 / 2 / 4 / 8-GPU training throughput scaling benchmark.

Launches `throughput_worker.py` via `torchrun` at each GPU count, parses the aggregate
tokens/sec it reports, and plots throughput + speedup vs. GPU count -- the concrete
answer to "how much do I actually gain from using more of this 8xB200 node for a given
model size?" Compare against lab 00's single-GPU baseline for the small (0.5B) model to
see how scaling efficiency differs by model size.

Run (needs all 8 GPUs free):
    python labs/07_capstone_scale_up/benchmark_scaling.py
    python labs/07_capstone_scale_up/benchmark_scaling.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common.paths import run_dir  # noqa: E402
from common.utils import banner  # noqa: E402

WORKER = os.path.join(os.path.dirname(__file__), "throughput_worker.py")
THROUGHPUT_RE = re.compile(r"THROUGHPUT_TOKENS_PER_SEC=([\d.]+)")


def run_worker(n_gpus: int, model: str) -> float:
    env = {**os.environ, "BENCH_MODEL": model}
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={n_gpus}", WORKER]
    print(f"\n$ {' '.join(cmd)}  (BENCH_MODEL={model})")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f"worker failed at n_gpus={n_gpus} (exit {result.returncode})")
    m = THROUGHPUT_RE.search(result.stdout)
    if not m:
        raise RuntimeError(f"could not parse throughput from worker output at n_gpus={n_gpus}")
    return float(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--gpu-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    import torch

    available = torch.cuda.device_count()
    gpu_counts = [n for n in args.gpu_counts if n <= available]
    if len(gpu_counts) < len(args.gpu_counts):
        print(f"[warn] only {available} GPU(s) visible; skipping counts above that.")

    banner(f"Lab 07: throughput scaling benchmark ({args.model})")
    results = {}
    for n in gpu_counts:
        results[n] = run_worker(n, args.model)

    baseline = results[gpu_counts[0]]
    print(f"\n{'GPUs':>5s} | {'tokens/sec':>12s} | {'speedup':>8s} | {'ideal speedup':>13s} | {'efficiency':>10s}")
    for n in gpu_counts:
        speedup = results[n] / baseline
        ideal = n / gpu_counts[0]
        print(f"{n:>5d} | {results[n]:>12,.0f} | {speedup:>8.2f} | {ideal:>13.2f} | {speedup / ideal:>9.1%}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(gpu_counts, [results[n] for n in gpu_counts], marker="o")
    ax1.set_xlabel("GPUs")
    ax1.set_ylabel("aggregate tokens/sec")
    ax1.set_title("Training throughput")
    ax1.grid(alpha=0.3)

    speedups = [results[n] / baseline for n in gpu_counts]
    ideal = [n / gpu_counts[0] for n in gpu_counts]
    ax2.plot(gpu_counts, speedups, marker="o", label="measured")
    ax2.plot(gpu_counts, ideal, linestyle="--", label="ideal (linear)")
    ax2.set_xlabel("GPUs")
    ax2.set_ylabel("speedup vs. smallest run")
    ax2.set_title("Scaling efficiency")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(args.model)
    fig.tight_layout()
    out_path = run_dir("lab07_capstone_scale_up") / f"scaling_{args.model.replace('/', '_')}.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
