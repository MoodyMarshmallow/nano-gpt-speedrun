#!/usr/bin/env bash
# Stage 0 baseline reproduction: run the ungated model multiple times.
# Usage:
#   scripts/run_baseline.sh               # run 3 seeds: 1337,1338,1339 on 8 GPUs
#   NPROC=4 COUNT=5 BASE_SEED=2000 scripts/run_baseline.sh
# Env vars:
#   NPROC       - number of GPUs per node (default 8)
#   COUNT       - how many runs (default 3)
#   BASE_SEED   - first seed; seeds increment by 1 per run (default 1337)
#   SCRIPT      - training script path (default train_gpt.py)
# Notes:
#   - Gating is forced off (ATTNGATE=none).
#   - Requires torchrun and the dataset at data/fineweb10B/fineweb_*.
#   - Does not modify code; only launches runs.

set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
else
  GPU_COUNT=0
fi

NPROC="${NPROC:-${GPU_COUNT:-8}}"
COUNT="${COUNT:-3}"
BASE_SEED="${BASE_SEED:-1337}"
SCRIPT="${SCRIPT:-train_gpt.py}"

if [ "${GPU_COUNT}" -gt 0 ] && [ "${NPROC}" -gt "${GPU_COUNT}" ]; then
  echo "Error: requested NPROC=${NPROC} but only ${GPU_COUNT} GPU(s) detected. Set NPROC <= ${GPU_COUNT} or adjust CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

for i in $(seq 0 $((COUNT-1))); do
  SEED=$((BASE_SEED + i))
  echo "==> Baseline run $((i+1))/$COUNT (seed=${SEED}, nproc=${NPROC})"
  if ! compgen -G "data/fineweb10B/fineweb_train_*.bin" > /dev/null; then
    echo "Error: no train shards found at data/fineweb10B/fineweb_train_*.bin." >&2
    echo "Ensure your data is in /workspace/fineweb10B and that data/fineweb10B -> /workspace/fineweb10B symlink exists." >&2
    exit 1
  fi
  if ! compgen -G "data/fineweb10B/fineweb_val_*.bin" > /dev/null; then
    echo "Error: no val shards found at data/fineweb10B/fineweb_val_*.bin." >&2
    exit 1
  fi
  ATTNGATE=none GATEPOS=sdpa GATEACT=sigmoid \
  SEED="${SEED}" \
  torchrun --standalone --nproc_per_node="${NPROC}" "${SCRIPT}"
done
