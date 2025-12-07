# Stage 1 Screening (Gating Ablations)

Goal: short runs (~1500 iters) to compare gating variants on the same hardware, stored under `/workspace/ese-3060-project`.

## 0) Prereqs
- Follow `docs/baseline_setup.md` to set up repo/env in `/workspace/ese-3060-project` and symlink data to `data/fineweb10B`.
- Activate venv: `cd /workspace/ese-3060-project && source .venv/bin/activate`.

---

## Option A — All-in-notebook (launch + EDA)
1) Start JupyterLab:
   ```bash
   cd /workspace/ese-3060-project
   source .venv/bin/activate
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''
   ```
2) Open `notebooks/stage1_runs_and_eda.ipynb`.
3) In “Launch runs”, set `DRY_RUN = False`, adjust `NPROC`, `NUM_ITER` (default 1500), and `SEEDS` (default 1337,1338).
4) Run the launch cell to start torchrun jobs for baseline + 4 gating variants (sdpa position).
5) Scroll down to the EDA sections to parse `experiments/results.csv` and `logs/*.txt` and view plots/tables.

Notes:
- `NPROC` must not exceed visible GPUs (auto-detected if unset).
- Logs: `logs/<run_id>.txt`; summary rows: `experiments/results.csv`.

---

## Option B — Shell script to run; notebook for EDA
1) Launch runs via script:
   ```bash
   cd /workspace/ese-3060-project
   SEEDS="1337 1338" NUM_ITER=1500 NPROC=4 scripts/run_stage1_screen.sh
   ```
   Configs (gate_pos=sdpa):
   - baseline (none)
   - headwise sigmoid
   - elementwise sigmoid
   - headwise ns_sigmoid (sparsity control)
   - const sigmoid (query-independent control)
2) Start JupyterLab (same as above) and open `notebooks/stage1_runs_and_eda.ipynb` — it will read existing results/logs and plot without launching new runs.

---

## 5) Quick validation checks
- Ensure each config has the expected number of seeds in the notebook summary table.
- If a run is missing, re-launch just that config/seed:
  ```bash
  SEEDS="2001" NPROC=4 NUM_ITER=1500 ATTNGATE=headwise GATEACT=sigmoid GATEPOS=sdpa torchrun --standalone --nproc_per_node=4 train_gpt.py
  ```
