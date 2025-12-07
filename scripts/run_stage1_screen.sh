#!/usr/bin/env bash
# Stage 1 screening runs (short, partial-length) for gating ablations.
# Uses 1500 iterations by default and runs 2 seeds per config.
# Everything assumes repo is at /workspace/ese-3060-project and data symlinked there.
#
# Env vars:
#   NPROC        - GPUs per node (auto-detected, default=all visible)
#   NUM_ITER     - iterations per run (default 1500)
#   SEEDS        - space-separated seeds (default "1337 1338")
#   SCRIPT       - training script (default train_gpt.py)
#
# Configs covered (gate_pos=sdpa):
#   - baseline (none)
#   - headwise sigmoid
#   - elementwise sigmoid
#   - headwise ns_sigmoid (sparsity control)
#   - const sigmoid (query-indep control)

set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
else
  GPU_COUNT=0
fi
NPROC="${NPROC:-${GPU_COUNT:-1}}"
NUM_ITER="${NUM_ITER:-1500}"
SEEDS="${SEEDS:-\"1337 1338\"}"
SCRIPT="${SCRIPT:-train_gpt.py}"

if [ "${GPU_COUNT}" -gt 0 ] && [ "${NPROC}" -gt "${GPU_COUNT}" ]; then
  echo "Error: requested NPROC=${NPROC} but only ${GPU_COUNT} GPU(s) detected." >&2
  exit 1
fi

declare -A CFGS
CFGS[baseline]="ATTNGATE=none GATEPOS=sdpa GATEACT=sigmoid"
CFGS[head_sig]="ATTNGATE=headwise GATEPOS=sdpa GATEACT=sigmoid"
CFGS[elem_sig]="ATTNGATE=elementwise GATEPOS=sdpa GATEACT=sigmoid"
CFGS[head_ns]="ATTNGATE=headwise GATEPOS=sdpa GATEACT=ns_sigmoid"
CFGS[const_sig]="ATTNGATE=const GATEPOS=sdpa GATEACT=sigmoid"

for cfg_name in baseline head_sig elem_sig head_ns const_sig; do
  eval CFG_STR=\"\${CFGS[$cfg_name]}\"
  echo "==> Config: ${cfg_name} (${CFG_STR}), iterations=${NUM_ITER}, seeds=${SEEDS}, nproc=${NPROC}"
  for seed in ${SEEDS}; do
    echo "----> Seed ${seed}"
    NUM_ITER=${NUM_ITER} VAL_EVERY=125 SEED=${seed} ${CFG_STR} \
      torchrun --standalone --nproc_per_node="${NPROC}" "${SCRIPT}"
  done
done
