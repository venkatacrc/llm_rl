"""Shared reward functions used across labs.

Three families:
  - `SentimentReward`      : a small pretrained classifier used as the reward model in
                              the classic "generate positive movie reviews" toy RLHF task
                              (labs 03, 08).
  - GSM8K / MATH verifiers : rule-based, *verifiable* rewards for the RLVR labs (05, 06,
                              07) -- no reward model needed, the environment just checks
                              if the final answer is correct.
  - `length_reward`        : a deliberately gameable proxy reward used to demonstrate
                              reward hacking / Goodhart's law in lab 08.

All reward functions follow the TRL `reward_funcs` calling convention:
    reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]
so they can be passed directly to `GRPOTrainer`/`RLOOTrainer`, or called directly in the
from-scratch scripts.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import torch


# --------------------------------------------------------------------------------------
# Sentiment reward (classic TRL "positive review" demo -- labs 03, 08)
# --------------------------------------------------------------------------------------


class SentimentReward:
    """Wraps a small sentiment classifier and returns the (logit or probability of the
    "positive" class) as a scalar reward. This is the exact reward used in the original
    TRL PPO-sentiment tutorial and in InstructGPT-style toy demos: cheap to evaluate
    (a distilled classifier, not an LLM judge), so you can run thousands of RL steps on
    a single GPU in minutes and *see* the policy shift its outputs.
    """

    def __init__(
        self,
        model_name: str = "lvwerra/distilbert-imdb",
        device: str | torch.device = "cuda",
        use_logit: bool = True,
        batch_size: int = 64,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.use_logit = use_logit
        self.batch_size = batch_size
        # lvwerra/distilbert-imdb: label 1 = POSITIVE
        self.positive_idx = 1

    @torch.no_grad()
    def __call__(self, prompts: list[str] | None = None, completions: list[str] | None = None, **kwargs: Any) -> list[float]:
        texts = completions if completions is not None else prompts
        assert texts is not None, "SentimentReward needs `completions` (or `prompts`)."
        rewards: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            chunk = [t if t.strip() else "." for t in chunk]  # classifier chokes on ""
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device
            )
            logits = self.model(**enc).logits
            if self.use_logit:
                scores = logits[:, self.positive_idx]
            else:
                scores = torch.softmax(logits, dim=-1)[:, self.positive_idx]
            rewards.extend(scores.float().cpu().tolist())
        return rewards


# --------------------------------------------------------------------------------------
# Verifiable rewards for math reasoning (RLVR -- labs 05, 06, 07)
# --------------------------------------------------------------------------------------

_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_gsm8k_gold_answer(reference_solution: str) -> str | None:
    """GSM8K gold solutions end with `#### <number>`."""
    m = _GSM8K_ANSWER_RE.search(reference_solution)
    return m.group(1).replace(",", "") if m else None


def _extract_final_number(text: str) -> str | None:
    """Best-effort extraction of a model's final numeric answer, checked in order of
    specificity: `#### x` (if the model was asked to mimic GSM8K format), `\\boxed{x}`
    (MATH-style), then just the last number to appear anywhere in the text.
    """
    m = _GSM8K_ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    m = _BOXED_RE.search(text[::-1])  # search from the end: take the *last* \boxed{}
    if m:
        # reverse back the captured (reversed) group
        return m.group(1)[::-1].replace(",", "").strip()
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _numeric_match(pred: str | None, gold: str | None, atol: float = 1e-4) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) <= atol
    except ValueError:
        return pred.strip() == gold.strip()


@lru_cache(maxsize=1)
def _try_import_math_verify():
    try:
        from math_verify import parse, verify  # type: ignore

        return parse, verify
    except Exception:
        return None, None


def math_correctness_reward(prompts: list[str], completions: list[str], gold_answers: list[str], **kwargs: Any) -> list[float]:
    """1.0 if the completion's final answer matches `gold_answers[i]`, else 0.0.

    Tries the `math-verify` package first (handles LaTeX/fraction/set equivalence
    properly, what serious RLVR pipelines use); falls back to a light regex + numeric
    comparison so this works even without that dependency installed.
    """
    parse, verify = _try_import_math_verify()
    rewards = []
    for completion, gold in zip(completions, gold_answers):
        if parse is not None:
            try:
                pred_parsed = parse(completion)
                gold_parsed = parse(gold if ("\\boxed" in gold or "$" in gold) else f"${gold}$")
                rewards.append(1.0 if verify(gold_parsed, pred_parsed) else 0.0)
                continue
            except Exception:
                pass  # fall through to the regex fallback for this example
        pred_num = _extract_final_number(completion)
        gold_num = _extract_final_number(gold) or gold
        rewards.append(1.0 if _numeric_match(pred_num, gold_num) else 0.0)
    return rewards


def format_reward(completions: list[str], tag_open: str = "<answer>", tag_close: str = "</answer>", **kwargs: Any) -> list[float]:
    """Small auxiliary reward that encourages the model to wrap its final answer in a
    tag (common trick in RLVR recipes so the verifier reward doesn't have to guess where
    the answer is). Returns 1.0 if the tags are present exactly once, else 0.0.
    """
    rewards = []
    for c in completions:
        ok = c.count(tag_open) == 1 and c.count(tag_close) == 1 and c.find(tag_open) < c.find(tag_close)
        rewards.append(1.0 if ok else 0.0)
    return rewards


# --------------------------------------------------------------------------------------
# Deliberately gameable reward (Goodhart's law demo -- lab 08)
# --------------------------------------------------------------------------------------


def length_reward(prompts: list[str] | None = None, completions: list[str] | None = None, target_tokens: int = 200, **kwargs: Any) -> list[float]:
    """A classic reward-hacking proxy: reward monotonically rewards longer completions
    (here, `+1` per ~4 characters, i.e. roughly per token, up to `target_tokens`, then
    flat). A policy optimized against *only* this reward will learn to ramble/pad
    instead of getting more helpful -- that's the point of lab 08.
    """
    assert completions is not None
    rewards = []
    for c in completions:
        approx_tokens = len(c) / 4.0
        rewards.append(min(approx_tokens, target_tokens) / target_tokens)
    return rewards


def reward_model_score(reward_model, reward_tokenizer, prompts: list[str], completions: list[str], device: str | torch.device = "cuda", batch_size: int = 32) -> list[float]:
    """Generic scorer for a trained sequence-classification reward model (the ones
    produced by lab 02), used as the "true" reward the proxy in lab 08 is compared
    against.
    """
    device = torch.device(device)
    reward_model.eval()
    scores: list[float] = []
    texts = [p + c for p, c in zip(prompts, completions)]
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = reward_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            out = reward_model(**enc).logits.squeeze(-1)
            scores.extend(out.float().cpu().tolist())
    return scores
