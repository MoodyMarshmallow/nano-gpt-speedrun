#!/usr/bin/env bash
# Stage 2 LR sweep: baseline vs elementwise gating, short runs.
# Configurable via env:
#   NPROC            - GPUs per run (default: auto-detect)
#   SEEDS            - space-separated seeds (default: "1337 2337")
#   LR_MULTS         - space-separated LR multipliers (default: "1.0 1.1 1.2 1.3")
#   BASE_LR          - base learning rate (default: 0.0036)
#   WARMUP_ITERS     - warmup steps (default: 150)
#   NUM_ITERATIONS   - total steps (default: 1000)
#   EARLY_STOP_PATIENCE - val-check patience before stopping (0 disables; default 2)
#   EARLY_STOP_MIN_DELTA - min improvement to reset patience (default 0.0)
#   SCRIPT           - path to train_gpt.py (default: /workspace/ese-3060-project/train_gpt.py)
# Notes:
#   - gate_pos=sdpa, gate_act=sigmoid
#   - baseline: attn_gate=none; elementwise: attn_gate=elementwise
#   - For divergence/NaN handling, rely on in-script early stopping.

set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
else
  GPU_COUNT=0
fi

NPROC="${NPROC:-${GPU_COUNT:-1}}"
SEEDS="${SEEDS:-1337 2337}"
LR_MULTS="${LR_MULTS:-1.0 1.1 1.2 1.3}"
BASE_LR="${BASE_LR:-0.0036}"
WARMUP_ITERS="${WARMUP_ITERS:-150}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
SCRIPT="${SCRIPT:-/workspace/ese-3060-project/train_gpt.py}"

if [ "${GPU_COUNT}" -gt 0 ] && [ "${NPROC}" -gt "${GPU_COUNT}" ]; then
  echo "Error: requested NPROC=${NPROC} but only ${GPU_COUNT} GPU(s) detected. Set NPROC <= ${GPU_COUNT} or adjust CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

run_one() {
  local attn_gate="$1"
  local lr="$2"
  local seed="$3"
  echo "==> attn_gate=${attn_gate} lr=${lr} seed=${seed} nproc=${NPROC}"
  ATTNGATE="${attn_gate}" GATEPOS=sdpa GATEACT=sigmoid \
  LR="${lr}" SEED="${seed}" \
  WARMUP_ITERS="${WARMUP_ITERS}" NUM_ITERATIONS="${NUM_ITERATIONS}" \
  EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE}" EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA}" \
  torchrun --standalone --nproc_per_node="${NPROC}" "${SCRIPT}"
}

for mult in ${LR_MULTS}; do
  lr=$(python3 - <<PY
base=${BASE_LR}
mult=${mult}
print(f"{base*mult:.10f}")
PY
)
  for seed in ${SEEDS}; do
    run_one "none" "${lr}" "${seed}"
  done
  for seed in ${SEEDS}; do
    run_one "elementwise" "${lr}" "${seed}"
  done
done
