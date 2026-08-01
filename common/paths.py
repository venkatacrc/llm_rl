"""Shared filesystem conventions so labs can hand checkpoints to each other.

Everything lands under `<repo_root>/checkpoints/<name>` and `<repo_root>/runs/<name>`
(metrics/plots), both git-ignored. Using `HF_HOME`'s parent (i.e. /raid) when available
keeps large checkpoints off the small root disk, matching `environment/setup.sh`.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _storage_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).resolve().parent  # e.g. /raid/llm_rl_cache
    return REPO_ROOT


CHECKPOINTS_DIR = _storage_root() / "checkpoints"
RUNS_DIR = _storage_root() / "runs"

# Well-known checkpoint names produced/consumed across labs.
SFT_MODEL_NAME = "sft_qwen2.5-0.5b_tldr"
REWARD_MODEL_NAME = "reward_model_qwen2.5-0.5b_tldr"


def checkpoint_dir(name: str, create: bool = True) -> Path:
    p = CHECKPOINTS_DIR / name
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(name: str, create: bool = True) -> Path:
    p = RUNS_DIR / name
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p
