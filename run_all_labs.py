#!/usr/bin/env python3
"""Run the LLM RL Alignment Labs end-to-end, in dependency order.

By default this runs every `scratch/` and `trl/` script for labs 00-06 and all of lab
08's scripts (single-GPU friendly, checkpoint hand-offs resolved automatically since
later scripts fall back to a base model with a warning if an upstream checkpoint is
missing). It **skips** two categories of step unless you opt in, since they're a much
bigger commitment than "work through a lab":

  - multi-GPU infra checks (lab 00's NCCL/8-GPU throughput bench, lab 07's scaling
    benchmark) -- quick (minutes), but need all 8 GPUs free.       --include-multi-gpu
  - the lab 07 capstone training run itself -- a real GRPO run on a 7-8B model across
    all 8 GPUs via FSDP, can run for a long time.                  --include-capstone

Examples:
    python run_all_labs.py --list                    # see every step without running anything
    python run_all_labs.py                            # run the default (single-GPU) curriculum
    python run_all_labs.py --only 00 01 02             # just those labs
    python run_all_labs.py --skip 05-trl 06-trl-rloo   # run everything except these steps
    python run_all_labs.py --include-multi-gpu --include-capstone   # the whole node, everything
    python run_all_labs.py --dry-run                  # print the commands without running them

Each step's combined stdout/stderr is streamed live and also saved to
`runs/run_all_labs/<step-id>.log`. A pass/fail/skip summary table is printed (and
written to `runs/run_all_labs/summary.csv`) at the end regardless of whether earlier
steps failed -- by default this script keeps going after a failure (some steps, like
the cross-checkpoint comparison/eval scripts, are expected to fail gracefully if you
haven't run every upstream script yet); pass `--fail-fast` to stop at the first error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from common.paths import run_dir  # noqa: E402


@dataclass
class Step:
    id: str
    lab: str
    description: str
    cmd: list[str]
    multi_gpu: bool = False  # needs several/all GPUs at once; excluded unless --include-multi-gpu
    capstone: bool = False  # the big lab-07 training run; excluded unless --include-capstone
    best_effort: bool = False  # expected to no-op/fail gracefully if upstream checkpoints are missing


def py(*parts: str) -> list[str]:
    return [sys.executable, str(REPO_ROOT.joinpath(*parts))]


def torchrun(n_gpus: int, *parts: str) -> list[str]:
    return ["torchrun", "--standalone", f"--nproc_per_node={n_gpus}", str(REPO_ROOT.joinpath(*parts))]


def accelerate_launch(config: str, *parts: str) -> list[str]:
    return [
        "accelerate",
        "launch",
        "--config_file",
        str(REPO_ROOT / "configs" / "accelerate" / config),
        str(REPO_ROOT.joinpath(*parts)),
    ]


STEPS: list[Step] = [
    # --- Lab 00: setup and GPU tour ---
    Step("00-check-env", "00", "Environment check", py("labs", "00_setup_and_gpu_tour", "check_env.py")),
    Step("00-hello-generate", "00", "Hello, generate", py("labs", "00_setup_and_gpu_tour", "hello_generate.py")),
    Step(
        "00-throughput-1gpu",
        "00",
        "Throughput benchmark (1 GPU)",
        py("labs", "00_setup_and_gpu_tour", "throughput_benchmark.py"),
    ),
    Step(
        "00-nccl-smoke-8gpu",
        "00",
        "NCCL all-reduce smoke test (8 GPU)",
        torchrun(8, "labs", "00_setup_and_gpu_tour", "nccl_smoke_test.py"),
        multi_gpu=True,
    ),
    Step(
        "00-throughput-8gpu",
        "00",
        "Throughput benchmark (8 GPU)",
        torchrun(8, "labs", "00_setup_and_gpu_tour", "throughput_benchmark.py"),
        multi_gpu=True,
    ),
    # --- Lab 01: SFT warmup ---
    Step("01-scratch", "01", "SFT warmup (scratch)", py("labs", "01_sft_warmup", "scratch", "train_sft_scratch.py")),
    Step("01-trl", "01", "SFT warmup (TRL)", py("labs", "01_sft_warmup", "trl", "train_sft_trl.py")),
    # --- Lab 02: reward modeling ---
    Step(
        "02-scratch",
        "02",
        "Reward modeling (scratch)",
        py("labs", "02_reward_modeling", "scratch", "train_rm_scratch.py"),
    ),
    Step("02-trl", "02", "Reward modeling (TRL)", py("labs", "02_reward_modeling", "trl", "train_rm_trl.py")),
    # --- Lab 03: PPO / classic RLHF ---
    Step("03-scratch", "03", "PPO RLHF (scratch)", py("labs", "03_ppo_rlhf", "scratch", "train_ppo_scratch.py")),
    Step(
        "03-trl",
        "03",
        "PPO RLHF (TRL)",
        py("labs", "03_ppo_rlhf", "trl", "train_ppo_trl.py"),
        best_effort=True,  # needs lab 02's reward model checkpoint
    ),
    # --- Lab 04: DPO family ---
    Step("04-scratch", "04", "DPO family (scratch)", py("labs", "04_dpo_family", "scratch", "train_dpo_scratch.py")),
    Step("04-trl-dpo", "04", "DPO (TRL)", py("labs", "04_dpo_family", "trl", "train_dpo_trl.py")),
    Step("04-trl-kto", "04", "KTO (TRL)", py("labs", "04_dpo_family", "trl", "train_kto_trl.py")),
    Step(
        "04-compare",
        "04",
        "PPO vs DPO vs KTO reward/KL comparison",
        py("labs", "04_dpo_family", "trl", "compare_ppo_dpo_kto.py"),
        best_effort=True,  # needs lab 01-04's TRL checkpoints to exist
    ),
    # --- Lab 05: GRPO and RLVR reasoning ---
    Step(
        "05-scratch",
        "05",
        "GRPO/RLVR (scratch)",
        py("labs", "05_grpo_rlvr_reasoning", "scratch", "train_grpo_scratch.py"),
    ),
    Step(
        "05-trl",
        "05",
        "GRPO/RLVR (TRL + vLLM)",
        py("labs", "05_grpo_rlvr_reasoning", "trl", "train_grpo_trl.py"),
    ),
    # --- Lab 06: RLOO and GSPO variants ---
    Step("06-scratch", "06", "RLOO (scratch)", py("labs", "06_rloo_and_gspo", "scratch", "train_rloo_scratch.py")),
    Step("06-trl-rloo", "06", "RLOO (TRL + vLLM)", py("labs", "06_rloo_and_gspo", "trl", "train_rloo_trl.py")),
    Step("06-trl-gspo", "06", "GSPO (TRL + vLLM)", py("labs", "06_rloo_and_gspo", "trl", "train_gspo_trl.py")),
    # --- Lab 07: capstone scale-up ---
    Step(
        "07-benchmark-scaling",
        "07",
        "1/2/4/8-GPU throughput scaling benchmark",
        py("labs", "07_capstone_scale_up", "benchmark_scaling.py"),
        multi_gpu=True,
    ),
    Step(
        "07-capstone-train",
        "07",
        "Capstone GRPO scale-up (7-8B model, FSDP, 8 GPU)",
        accelerate_launch("fsdp_8gpu.yaml", "labs", "07_capstone_scale_up", "train_grpo_capstone.py"),
        multi_gpu=True,
        capstone=True,
    ),
    # --- Lab 08: safety and reward hacking ---
    Step(
        "08-reward-hacking",
        "08",
        "Reward hacking demo (kl_beta=0.0)",
        py("labs", "08_safety_and_reward_hacking", "reward_hacking_demo.py"),
    ),
    Step(
        "08-kl-frontier",
        "08",
        "KL-reward frontier sweep",
        py("labs", "08_safety_and_reward_hacking", "kl_reward_frontier.py"),
    ),
    Step(
        "08-alignment-tax",
        "08",
        "Alignment tax (MMLU)",
        py("labs", "08_safety_and_reward_hacking", "alignment_tax.py"),
        best_effort=True,  # needs at least the lab-01 SFT checkpoint
    ),
    Step(
        "08-redteam",
        "08",
        "Red-team refusal-rate eval",
        py("labs", "08_safety_and_reward_hacking", "redteam_eval.py"),
        best_effort=True,  # needs at least one trained checkpoint; downloads a judge model
    ),
]


def default_enabled(step: Step, args: argparse.Namespace) -> bool:
    if step.capstone and not args.include_capstone:
        return False
    if step.multi_gpu and not args.include_multi_gpu:
        return False
    return True


def print_table(steps: list[Step]) -> None:
    id_w = max(len(s.id) for s in steps)
    print(f"{'step id'.ljust(id_w)}  lab  flags        description")
    for s in steps:
        flags = "+".join(f for f, on in [("multi-gpu", s.multi_gpu), ("capstone", s.capstone)] if on) or "-"
        print(f"{s.id.ljust(id_w)}  {s.lab:>3s}  {flags:<12s} {s.description}")


def tee_run(step: Step, log_path: Path) -> tuple[int, float]:
    print(f"\n{'=' * 70}\n>>> [{step.id}] {step.description}\n$ {' '.join(step.cmd)}\n{'=' * 70}")
    t0 = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        try:
            proc = subprocess.Popen(
                step.cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
        except FileNotFoundError as e:
            print(f"[{step.id}] failed to launch: {e}")
            logf.write(str(e))
            return 127, time.perf_counter() - t0
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    return proc.returncode, time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print all steps (with default enabled/disabled) and exit")
    parser.add_argument("--only", nargs="+", metavar="LAB", help="only run steps for these lab numbers, e.g. 00 01 02")
    parser.add_argument("--steps", nargs="+", metavar="STEP_ID", help="only run these specific step ids")
    parser.add_argument("--skip", nargs="+", metavar="STEP_ID", default=[], help="exclude these specific step ids")
    parser.add_argument(
        "--include-multi-gpu", action="store_true", help="also run steps that need several/all 8 GPUs at once"
    )
    parser.add_argument(
        "--include-capstone", action="store_true", help="also run lab 07's full capstone training (long-running)"
    )
    parser.add_argument("--fail-fast", action="store_true", help="stop at the first failing step")
    parser.add_argument("--dry-run", action="store_true", help="print the commands that would run, without running them")
    args = parser.parse_args()

    if args.list:
        print_table(STEPS)
        return 0

    selected = STEPS
    if args.only:
        wanted = {lab.zfill(2) for lab in args.only}
        selected = [s for s in selected if s.lab in wanted]
    if args.steps:
        wanted_ids = set(args.steps)
        selected = [s for s in selected if s.id in wanted_ids]
    selected = [s for s in selected if s.id not in set(args.skip)]

    to_run, to_skip = [], []
    for s in selected:
        (to_run if default_enabled(s, args) else to_skip).append(s)

    print(f"LLM RL Alignment Labs -- running {len(to_run)} step(s), skipping {len(to_skip)}.")
    if to_skip:
        print("Skipped (pass --include-multi-gpu / --include-capstone, or --steps, to include):")
        for s in to_skip:
            print(f"  - [{s.id}] {s.description}")

    if args.dry_run:
        print("\n--dry-run: commands that would execute, in order:")
        for s in to_run:
            print(f"  [{s.id}] $ {' '.join(s.cmd)}")
        return 0

    log_dir = run_dir("run_all_labs")
    results: list[tuple[Step, str, float]] = []  # (step, status, duration_s)

    try:
        for step in to_run:
            log_path = log_dir / f"{step.id}.log"
            code, duration = tee_run(step, log_path)
            if code == 0:
                status = "OK"
            elif step.best_effort:
                status = "SKIPPED (best-effort, see log)"
            else:
                status = "FAILED"
            results.append((step, status, duration))
            print(f"<<< [{step.id}] {status} in {duration:.1f}s (log: {log_path})")
            if status == "FAILED" and args.fail_fast:
                print("\n--fail-fast set: stopping here.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user -- printing partial summary below.")

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    id_w = max((len(s.id) for s, _, _ in results), default=8)
    summary_csv = log_dir / "summary.csv"
    with open(summary_csv, "w") as f:
        f.write("step_id,description,status,duration_s\n")
        for step, status, duration in results:
            print(f"{step.id.ljust(id_w)}  {status:<28s} {duration:>7.1f}s   {step.description}")
            f.write(f"{step.id},{step.description},{status},{duration:.1f}\n")
    print(f"\nFull logs: {log_dir}/<step-id>.log")
    print(f"Summary CSV: {summary_csv}")

    n_failed = sum(1 for _, status, _ in results if status == "FAILED")
    if n_failed:
        print(f"\n{n_failed} step(s) failed (not counting best-effort skips). Check the logs above.")
        return 1
    print("\nAll executed steps completed (or gracefully skipped as best-effort).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
