#!/usr/bin/env python3
"""Split experiments/results.csv into stage-specific CSVs.

Defaults:
  - Stage 1: num_iterations == 1500 (screening runs)
  - Stage 2: num_iterations == 800 (LR sweep runs)

Usage:
  python scripts/split_results.py \
    --source experiments/results.csv \
    --stage1-iters 1500 \
    --stage2-iters 800 \
    --out-dir experiments

You can override the iteration sets with comma-separated lists.
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="experiments/results.csv", help="Input aggregated results CSV")
    p.add_argument("--stage1-iters", default="1500", help="Comma-separated num_iterations values for stage 1")
    p.add_argument("--stage2-iters", default="800", help="Comma-separated num_iterations values for stage 2")
    p.add_argument("--out-dir", default="experiments", help="Directory to write stage CSVs")
    return p.parse_args()


def parse_iter_list(s: str) -> set[int]:
    return {int(x) for x in s.split(',') if x.strip()}


def main():
    args = parse_args()
    src = Path(args.source)
    out_dir = Path(args.out_dir)
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    df = pd.read_csv(src)
    stage1_iters = parse_iter_list(args.stage1_iters)
    stage2_iters = parse_iter_list(args.stage2_iters)

    stage1 = df[df.get("num_iterations", 0).isin(stage1_iters)].copy()
    stage2 = df[df.get("num_iterations", 0).isin(stage2_iters)].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = out_dir / "results_stage1.csv"
    stage2_path = out_dir / "results_stage2.csv"
    stage1.to_csv(stage1_path, index=False)
    stage2.to_csv(stage2_path, index=False)

    print(f"Wrote stage1 ({len(stage1)}) -> {stage1_path}")
    print(f"Wrote stage2 ({len(stage2)}) -> {stage2_path}")


if __name__ == "__main__":
    main()
