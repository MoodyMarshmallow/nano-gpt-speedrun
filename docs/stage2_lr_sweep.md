# Stage 2 LR Sweep (Baseline vs Elementwise)

Goal: run LR multipliers for baseline (no gate) and elementwise SDPA-output gating, with early-stop on divergence. Short runs: 1,000 iters, 150 warmup iters, 2 seeds per LR by default.

## 0) Prereqs
- Follow `docs/baseline_setup.md` to set up repo/env in `/workspace/ese-3060-project` and symlink data to `data/fineweb10B`.
- Ensure `train_gpt.py` is from this repo version (env overrides for `WARMUP_ITERS`/`NUM_ITERATIONS` are supported).
- Activate venv: `cd /workspace/ese-3060-project && source .venv/bin/activate`.

## Option A — Notebook (launch + plots)
1) Start JupyterLab:
   ```bash
   cd /workspace/ese-3060-project
   source .venv/bin/activate
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''
   ```
2) Open `notebooks/stage2_lr_sweep.ipynb`.
3) In the config cell, set:
   - `gpus_per_run` to your GPU count (e.g., 1 or 8)
   - `cuda_visible_devices` if you want to pin devices (or leave None)
   - `lr_multipliers` (default [1.0, 1.1, 1.2, 1.3])
   - `seeds` (default [1337, 2337])
   - `warmup_iters=150`, `num_iterations=1000` (already set)
4) Remove the `if False` guard in the run cell to launch. The notebook watches stdout for NaN/inf or loss >10 and terminates that run early.
5) After runs, execute the plotting cell at the end. It parses `logs/*.txt`, averages val loss per step across seeds, and plots baseline vs elementwise with LRs overlaid.

Notes:
- Runs use env overrides: `ATTNGATE` (`none` or `elementwise`), `GATEPOS=sdpa`, `GATEACT=sigmoid`, `LR`, `SEED`, `WARMUP_ITERS=150`, `NUM_ITERATIONS=1000`.
- Logs go to `logs/<run_id>.txt`; `experiments/results.csv` is appended per run.

## Option B — Shell (manual torchrun) + notebook for plots
1) Launch runs manually (example for 1 GPU):
   ```bash
   cd /workspace/ese-3060-project
   # Baseline
   LR=0.0036 SEED=1337 WARMUP_ITERS=150 NUM_ITERATIONS=1000 ATTNGATE=none GATEPOS=sdpa GATEACT=sigmoid \
   torchrun --standalone --nproc_per_node=1 train_gpt.py
   # Elementwise with 1.2x LR
   LR=0.0036*1.2 SEED=2337 WARMUP_ITERS=150 NUM_ITERATIONS=1000 ATTNGATE=elementwise GATEPOS=sdpa GATEACT=sigmoid \
   torchrun --standalone --nproc_per_node=1 train_gpt.py
   ```
   Repeat for the LR multipliers/seed grid you want (e.g., 1.0, 1.1, 1.2, 1.3 with 2 seeds).
2) Open `notebooks/stage2_lr_sweep.ipynb` and run the final plot cell (you can leave the run cell disabled); it will read existing logs and plot curves.

## Quick checks
- Ensure `CUDA_VISIBLE_DEVICES` exposes enough GPUs for `nproc_per_node`.
- Verify `warmup_iters=150`, `num_iterations=1000` in logs match the sweep settings; the plot cell filters on these.
- If a run diverges and stops early, its log will be truncated; rerun that LR/seed if needed.
