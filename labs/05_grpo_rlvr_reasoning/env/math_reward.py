"""Lab 05 environment: verifiable-reward helpers for GSM8K/MATH.

Thin, lab-specific wrapper around `common/rewards.py`'s math verifier, plus GSM8K
dataset loading/formatting shared by the scratch and TRL scripts in this lab.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from common.rewards import extract_gsm8k_gold_answer, format_reward, math_correctness_reward  # noqa: E402,F401

GSM8K_SYSTEM_PROMPT = (
    "Solve the math problem. Show your reasoning briefly, then give the final answer "
    "on its own line as: #### <number>"
)


def load_gsm8k(split: str = "train"):
    """Returns a `datasets.Dataset` with columns `prompt` (chat-formatted question) and
    `gold_answer` (the numeric answer extracted from GSM8K's `#### x` gold solution).
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)

    def to_prompt(example):
        gold = extract_gsm8k_gold_answer(example["answer"])
        return {
            "prompt": [
                {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "gold_answer": gold,
        }

    return ds.map(to_prompt, remove_columns=ds.column_names)


def gsm8k_reward(prompts, completions, gold_answer, **kwargs):
    """TRL `reward_funcs`-compatible: `math_correctness_reward` plus a light format
    bonus for using the `#### <number>` convention (helps the parser and, empirically,
    speeds up convergence in RLVR training).
    """
    correctness = math_correctness_reward(prompts, completions, gold_answer)
    fmt = [1.0 if "####" in c else 0.0 for c in completions]
    return [c + 0.1 * f for c, f in zip(correctness, fmt)]
