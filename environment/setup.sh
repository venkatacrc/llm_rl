#!/usr/bin/env bash
# Environment setup for the LLM RL Alignment Labs on an 8xB200 node.
#
# What this does:
#   1. Creates a dedicated conda env `llm_rl` (does NOT touch `base`).
#   2. Points all HF/torch/pip caches at /raid (root disk is small, /raid is huge).
#   3. Installs the pinned requirements + a CUDA-13/Blackwell-compatible torch build.
#   4. Runs the lab 00 environment check so you know immediately if something's off.
#
# Usage:
#   bash environment/setup.sh
#
# Re-running is safe/idempotent (conda env create is skipped if it already exists).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="llm_rl"
CACHE_ROOT="/raid/llm_rl_cache"

echo "== LLM RL Alignment Labs: environment setup =="
echo "Repo root: ${REPO_ROOT}"

# --- 1. Cache dirs on /raid (root disk only has ~1.5TB free; /raid has ~27TB) ---
if [ -d /raid ]; then
  echo "-- Using /raid for model/dataset/pip caches"
  mkdir -p "${CACHE_ROOT}/hf_home" "${CACHE_ROOT}/hf_datasets" "${CACHE_ROOT}/torch" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/vllm"
else
  echo "-- /raid not found, falling back to \$HOME for caches (check disk space with 'df -h')"
  CACHE_ROOT="${HOME}/.cache/llm_rl_cache"
  mkdir -p "${CACHE_ROOT}/hf_home" "${CACHE_ROOT}/hf_datasets" "${CACHE_ROOT}/torch" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/vllm"
fi

ENV_FILE="${REPO_ROOT}/environment/env.sh"
cat > "${ENV_FILE}" <<EOF
# Source this file in every shell you use for the labs:
#   source environment/env.sh
export HF_HOME="${CACHE_ROOT}/hf_home"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_datasets"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TOKENIZERS_PARALLELISM=false
# Uncomment to enable Weights & Biases instead of the default local CSV/trackio logging:
# export WANDB_PROJECT=llm_rl_alignment_labs
EOF
echo "-- Wrote ${ENV_FILE} (cache dirs under ${CACHE_ROOT})"
# shellcheck disable=SC1090
source "${ENV_FILE}"

# --- 2. Conda env ---
if ! command -v conda &>/dev/null; then
  echo "conda not found on PATH. Install/activate conda first (this node has a (base) conda env)." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "-- Conda env '${ENV_NAME}' already exists, activating it"
else
  echo "-- Creating conda env '${ENV_NAME}' (python 3.11)"
  conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"

# --- 3. Torch for CUDA 13.0 / Blackwell, then the rest of requirements.txt ---
echo "-- Installing torch (CUDA 12.8 wheels; forward-compatible with driver's CUDA 13.0 runtime)"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "-- Installing remaining requirements"
pip install -r "${REPO_ROOT}/requirements.txt"

echo "-- Installing flash-attn (optional, speeds up labs 05/06/07; skip on failure)"
pip install flash-attn --no-build-isolation || echo "   (flash-attn build failed, continuing without it -- labs fall back to sdpa attention)"

# --- 4. Sanity check ---
echo "== Running environment check (labs/00_setup_and_gpu_tour/check_env.py) =="
python "${REPO_ROOT}/labs/00_setup_and_gpu_tour/check_env.py" || true

echo ""
echo "Setup done. For every new shell, run:"
echo "  conda activate ${ENV_NAME}"
echo "  source ${REPO_ROOT}/environment/env.sh"
