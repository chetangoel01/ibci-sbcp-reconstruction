"""Ensemble submission CSVs by weighted averaging.

Usage:
    # Equal-weight ensemble of all available submissions:
    python ensemble.py

    # Specify files and weights explicitly:
    python ensemble.py \
        --submissions sub_transformer.csv sub_gru.csv \
        --weights 0.6 0.4

    # Weight by val R² (default behavior when no weights given):
    python ensemble.py --submissions sub1.csv sub2.csv sub3.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POSITION_COLS = ["index_pos", "mrp_pos"]


def ensemble(
    submissions: list[Path],
    weights: list[float] | None = None,
    output: Path = Path("submission_ensemble.csv"),
) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in submissions]

    n = len(dfs)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    ref = dfs[0].copy()
    for col in POSITION_COLS:
        ref[col] = sum(w * df[col].values for w, df in zip(weights, dfs))

    ref[POSITION_COLS] = ref[POSITION_COLS].clip(0.0, 1.0)

    ref.to_csv(output, index=False)
    print(f"Ensemble saved: {output} ({len(ref)} rows)")
    print(f"  Inputs: {[p.name for p in submissions]}")
    print(f"  Weights: {[f'{w:.3f}' for w in weights]}")
    return ref


def main():
    parser = argparse.ArgumentParser(description="Ensemble submission CSVs")
    parser.add_argument(
        "--submissions", nargs="+", type=str, default=None,
        help="Paths to submission CSVs. If omitted, auto-discovers from phase2_outputs/results/",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Weights per submission (normalized automatically). If omitted, equal weights.",
    )
    parser.add_argument(
        "--output", type=str, default="submission_ensemble.csv",
        help="Output path for ensemble submission",
    )
    args = parser.parse_args()

    if args.submissions:
        paths = [Path(s) for s in args.submissions]
    else:
        results_dir = Path("phase2_outputs/results")
        paths = sorted(results_dir.glob("submission_*.csv"))
        if not paths:
            print("No submission CSVs found in phase2_outputs/results/")
            return

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    print(f"Ensembling {len(paths)} submissions...")
    ensemble(paths, weights=args.weights, output=Path(args.output))


if __name__ == "__main__":
    main()
