"""Shared utilities for the LLM RL Alignment Labs.

Kept dependency-light on purpose: every lab should run with nothing more than the
packages in requirements.txt, and without needing a Weights & Biases account. Metrics
are always logged to a plain CSV (so you can plot/inspect with anything), with an
optional CSV -> PNG plot helper, and an optional W&B backend if you've set WANDB_PROJECT.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Reproducibility / device
# --------------------------------------------------------------------------------------


def set_seed(seed: int = 0) -> None:
    """Seed python/numpy/torch (+ CUDA) RNGs. Call this first thing in every script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - not the target HW
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
    return sum(p.numel() for p in params)


def human_count(n: int) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else str(n)
        n /= 1000.0
    return f"{n:.1f}P"


# --------------------------------------------------------------------------------------
# Lightweight run logger: CSV always, W&B optionally, PNG plot helper
# --------------------------------------------------------------------------------------


class RunLogger:
    """Append scalar metrics to `<run_dir>/metrics.csv`, print them, and optionally
    mirror to Weights & Biases if `WANDB_PROJECT` is set in the environment and `wandb`
    is importable. Call `.plot(...)` at the end (or anytime) to render a PNG.
    """

    def __init__(self, run_dir: str | Path, run_name: str, config: dict[str, Any] | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.csv_path = self.run_dir / "metrics.csv"
        self._fieldnames: list[str] | None = None
        self._rows: list[dict[str, Any]] = []
        self._t0 = time.time()

        if config is not None:
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

        self._wandb = None
        if os.environ.get("WANDB_PROJECT"):
            try:
                import wandb  # type: ignore

                self._wandb = wandb.init(
                    project=os.environ["WANDB_PROJECT"], name=run_name, config=config, reinit=True
                )
            except Exception as e:  # pragma: no cover - optional dependency
                print(f"[RunLogger] wandb logging disabled ({e})")

    def log(self, step: int, **metrics: float) -> None:
        row = {"step": step, "wall_time_s": round(time.time() - self._t0, 2), **metrics}
        self._rows.append(row)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            write_header = True
        else:
            write_header = False
            for k in row:
                if k not in self._fieldnames:
                    self._fieldnames.append(k)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header and f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

        pretty = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"[{self.run_name}] step {step:>5d} | {pretty}")

        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def plot(self, keys: Iterable[str], out_name: str = "training_curves.png", title: str | None = None) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = [k for k in keys if any(k in r for r in self._rows)]
        if not keys:
            raise ValueError("No matching keys logged yet; nothing to plot.")

        fig, axes = plt.subplots(len(keys), 1, figsize=(7, 3 * len(keys)), sharex=True)
        if len(keys) == 1:
            axes = [axes]
        steps = [r["step"] for r in self._rows]
        for ax, key in zip(axes, keys):
            ys = [r.get(key) for r in self._rows]
            xs = [s for s, y in zip(steps, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            ax.plot(xs, ys)
            ax.set_ylabel(key)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("step")
        fig.suptitle(title or self.run_name)
        fig.tight_layout()
        out_path = self.run_dir / out_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[{self.run_name}] wrote plot -> {out_path}")
        return out_path

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


# --------------------------------------------------------------------------------------
# Generation helpers (used by every "scratch" RL script for rollouts)
# --------------------------------------------------------------------------------------


@dataclass
class GenerationBatch:
    prompt_ids: torch.Tensor  # (B, P) left-padded prompt token ids
    prompt_mask: torch.Tensor  # (B, P)
    response_ids: torch.Tensor  # (B, R) right-padded response token ids
    response_mask: torch.Tensor  # (B, R), 1 for real tokens (incl. EOS), 0 for padding
    full_ids: torch.Tensor  # (B, P+R) = cat(prompt_ids, response_ids)
    full_mask: torch.Tensor  # (B, P+R)


@torch.no_grad()
def sample_responses(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
) -> GenerationBatch:
    """Thin wrapper around `generate` that also returns clean prompt/response masks,
    which every from-scratch PPO/GRPO/RLOO script needs to compute per-token log-probs.
    """
    model.eval()
    device = prompt_ids.device
    pad_id = tokenizer.pad_token_id
    out = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = prompt_ids.shape[1]
    response_ids = out[:, prompt_len:]

    # Response mask is 1 up to and including the first EOS token, 0 after (covers the
    # common case where pad_token == eos_token too). If the model never emits EOS
    # within max_new_tokens, every generated token counts.
    seq_len = response_ids.shape[1]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        is_eos = response_ids == eos_id
        no_eos = ~is_eos.any(dim=1)
        first_eos = is_eos.float().argmax(dim=1)
        first_eos = torch.where(no_eos, torch.full_like(first_eos, seq_len - 1), first_eos)
        positions = torch.arange(seq_len, device=response_ids.device).unsqueeze(0)
        response_mask = (positions <= first_eos.unsqueeze(1)).long()
    else:
        response_mask = torch.ones_like(response_ids)

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask], dim=1)
    return GenerationBatch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        response_ids=response_ids,
        response_mask=response_mask,
        full_ids=full_ids,
        full_mask=full_mask,
    )


def decode_batch(tokenizer, ids: torch.Tensor, mask: torch.Tensor | None = None) -> list[str]:
    texts = []
    for i in range(ids.shape[0]):
        row = ids[i]
        if mask is not None:
            row = row[mask[i].bool()]
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def left_pad_batch(tokenizer, texts: list[str], device: torch.device, max_length: int | None = None):
    """Tokenize with left-padding, which is what you want for batched `generate()`."""
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    tokenizer.padding_side = old_side
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def banner(msg: str) -> None:
    line = "=" * max(len(msg) + 4, 40)
    print(f"\n{line}\n  {msg}\n{line}\n")
