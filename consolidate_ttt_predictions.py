"""Consolidate per-session TTT prediction CSVs into `results/mae_predictions.csv`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import ensure_output_dirs, get_config, preflight_validate_submission_indices, setup_logging
from data_utils import load_test_mask_csv

LOGGER = logging.getLogger(__name__)


REQUIRED_COLS = {"sample_id", "session_id", "time_bin", "channel", "predicted_sbp"}



def consolidate_ttt_predictions(config_name: str = "hpc", strict: bool = True) -> Path:
    """Consolidate `results/ttt_predictions/*.csv` into `results/mae_predictions.csv`.

    Args:
        config_name: Config profile (`local` or `hpc`).
        strict: If True, require full test-mask coverage.

    Returns:
        Path to the consolidated output CSV.
    """
    config = get_config(config_name)
    ensure_output_dirs(config)
    preflight_validate_submission_indices(config)

    pred_dir = config.ttt_predictions_dir
    if not pred_dir.exists():
        raise FileNotFoundError(f"TTT predictions directory does not exist: {pred_dir}")

    files = sorted(pred_dir.glob("S*.csv"))
    if not files:
        raise FileNotFoundError(f"No per-session TTT prediction CSVs found in {pred_dir}")

    LOGGER.info("Consolidating %d TTT prediction files from %s", len(files), pred_dir)
    parts: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if df["predicted_sbp"].isna().any():
            raise ValueError(f"{path} contains NaN predictions")
        if not np.isfinite(df["predicted_sbp"].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{path} contains non-finite predictions")
        parts.append(df[["sample_id", "session_id", "time_bin", "channel", "predicted_sbp"]].copy())

    merged = pd.concat(parts, axis=0, ignore_index=True)
    if merged["sample_id"].duplicated().any():
        dups = int(merged["sample_id"].duplicated().sum())
        raise ValueError(f"Consolidated TTT predictions contain duplicate sample_id values ({dups})")

    test_mask_df = load_test_mask_csv(config)
    if strict:
        aligned = test_mask_df.merge(merged[["sample_id", "predicted_sbp"]], on="sample_id", how="left")
        if aligned["predicted_sbp"].isna().any():
            n_missing = int(aligned["predicted_sbp"].isna().sum())
            raise ValueError(f"TTT predictions are missing {n_missing} sample IDs")
        out_df = aligned[["sample_id", "session_id", "time_bin", "channel", "predicted_sbp"]].copy()
    else:
        out_df = merged.sort_values("sample_id").reset_index(drop=True)

    out_path = config.results_dir / "mae_predictions.csv"
    out_df.to_csv(out_path, index=False)
    LOGGER.info("Wrote consolidated MAE predictions to %s (%d rows)", out_path, len(out_df))
    return out_path



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate per-session TTT prediction CSVs")
    parser.add_argument("--config", choices=["local", "hpc"], default="hpc")
    parser.add_argument("--no_strict", action="store_true", help="Do not enforce full test-mask coverage")
    return parser.parse_args()



def main() -> None:
    args = _parse_args()
    config = get_config(args.config)
    ensure_output_dirs(config)
    setup_logging(config.log_level, config.logs_dir / "consolidate_ttt_predictions.log")
    consolidate_ttt_predictions(config_name=args.config, strict=not args.no_strict)


if __name__ == "__main__":
    main()
