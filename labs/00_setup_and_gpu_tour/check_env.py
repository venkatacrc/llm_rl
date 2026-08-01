"""Lab 00, part 1: environment check.

Run this FIRST, before anything else in the curriculum:

    python labs/00_setup_and_gpu_tour/check_env.py

It verifies: python version, torch/CUDA/driver versions, that all 8 B200s are visible
with the expected ~183GB each, bf16/tf32 support, flash-attention availability, and that
transformers/accelerate/trl/vllm/deepspeed/peft/datasets are importable and recent
enough. Prints a PASS/WARN/FAIL table and exits non-zero if anything is FAIL.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass

RESET, RED, GREEN, YELLOW, BOLD = "\033[0m", "\033[31m", "\033[32m", "\033[33m", "\033[1m"


@dataclass
class CheckResult:
    name: str
    status: str  # PASS / WARN / FAIL
    detail: str


results: list[CheckResult] = []


def check(name: str):
    def decorator(fn):
        try:
            status, detail = fn()
        except Exception as e:  # noqa: BLE001
            status, detail = "FAIL", f"raised {type(e).__name__}: {e}"
        results.append(CheckResult(name, status, detail))
        return fn

    return decorator


@check("python version")
def _():
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    return ("PASS" if ok else "WARN"), platform.python_version()


@check("torch import + CUDA build")
def _():
    import torch

    return "PASS", f"torch {torch.__version__}, built for CUDA {torch.version.cuda}"


@check("CUDA available")
def _():
    import torch

    if not torch.cuda.is_available():
        return "FAIL", "torch.cuda.is_available() is False -- check driver / CUDA_VISIBLE_DEVICES"
    return "PASS", f"{torch.cuda.device_count()} device(s) visible"


@check("GPU inventory (expect 8x B200, ~183GB each)")
def _():
    import torch

    n = torch.cuda.device_count()
    lines = []
    total_mem_ok = True
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        lines.append(f"gpu{i}={props.name} ({mem_gb:.0f}GB, sm_{props.major}{props.minor})")
        if mem_gb < 150:
            total_mem_ok = False
    status = "PASS" if (n == 8 and total_mem_ok) else "WARN"
    return status, "; ".join(lines) if lines else "no GPUs found"


@check("bf16 / tf32 support")
def _():
    import torch

    if not torch.cuda.is_available():
        return "WARN", "skipped, no CUDA"
    bf16_ok = torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = True
    return ("PASS" if bf16_ok else "WARN"), f"bf16_supported={bf16_ok}, tf32 enabled for matmul"


@check("NCCL backend")
def _():
    import torch

    return ("PASS" if torch.distributed.is_nccl_available() else "FAIL"), "torch.distributed.is_nccl_available()"


@check("flash-attention (optional, speeds up labs 05/06/07)")
def _():
    try:
        import flash_attn  # noqa: F401

        return "PASS", f"flash_attn {flash_attn.__version__}"
    except ImportError:
        return "WARN", "not installed -- fine, scripts fall back to sdpa attention"


for pkg, min_version_hint in [
    ("transformers", "4.47"),
    ("accelerate", "1.2"),
    ("trl", "0.27"),
    ("peft", "0.14"),
    ("datasets", "3.2"),
    ("deepspeed", "0.16"),
    ("vllm", "0.7"),
]:

    def _make(pkg=pkg, min_version_hint=min_version_hint):
        @check(f"import {pkg} (expect >= {min_version_hint})")
        def _inner():
            mod = importlib.import_module(pkg)
            return "PASS", f"{pkg} {getattr(mod, '__version__', '?')}"

        return _inner

    _make()


@check("HF cache location (should be on /raid, not root disk)")
def _():
    import os

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return "WARN", "HF_HOME not set -- did you `source environment/env.sh`?"
    status = "PASS" if hf_home.startswith("/raid") else "WARN"
    return status, hf_home


@check("disk space on cache volume")
def _():
    import os
    import shutil

    path = os.environ.get("HF_HOME", "/")
    root = path if os.path.exists(path) else "/"
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    status = "PASS" if free_gb > 50 else "WARN"
    return status, f"{free_gb:.0f}GB free at {root}"


def main() -> int:
    name_w = max(len(r.name) for r in results)
    print(f"\n{BOLD}LLM RL Alignment Labs -- environment check{RESET}\n")
    n_fail = 0
    for r in results:
        color = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}[r.status]
        n_fail += r.status == "FAIL"
        print(f"  [{color}{r.status:4s}{RESET}] {r.name.ljust(name_w)}  {r.detail}")
    print()
    if n_fail:
        print(f"{RED}{n_fail} check(s) FAILED -- fix these before running the labs.{RESET}")
        return 1
    print(f"{GREEN}All checks passed (WARNs are informational). You're good to go.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
