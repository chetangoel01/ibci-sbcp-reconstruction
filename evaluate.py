"""Local evaluation and analysis utilities for iBCI SBP reconstruction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import metric
from config import Config, ensure_output_dirs, get_config, preflight_validate_submission_indices, setup_logging
from data_utils import (
    assign_trial_indices,
    build_pseudo_solution_df,
    build_submission_like_df_from_dense,
    create_artificial_mask,
    load_metadata,
    load_test_mask_csv,
    load_train_session,
    zscore_session,
)
from pillar1_gaussian import ConditionalGaussian
from pillar2_mae_ttt import load_pretrained_model, prepare_test_session_adaptation_data
from pillar2_mae_train import _gather_windows_numpy

LOGGER = logging.getLogger(__name__)



def _pseudo_test_mask_from_train_session(
    sbp: np.ndarray,
    trial_starts: np.ndarray,
    trial_ends: np.ndarray,
    n_masked_trials: int,
    n_channels_to_mask: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a pseudo-test mask by masking whole trials with constant channel sets per trial."""
    rng = np.random.default_rng(seed)
    n_trials = len(trial_starts)
    choose = min(n_masked_trials, n_trials)
    masked_trial_idx = np.sort(rng.choice(n_trials, size=choose, replace=False))
    mask = np.zeros_like(sbp, dtype=bool)
    for i in masked_trial_idx:
        s = int(trial_starts[i])
        e = int(trial_ends[i])
        ch = rng.choice(sbp.shape[1], size=n_channels_to_mask, replace=False)
        mask[s:e, ch] = True
    masked = sbp.copy()
    masked[mask] = 0.0
    return masked.astype(np.float32), mask



def cross_validate_gaussian(config: Config, n_folds: int = 5) -> pd.DataFrame:
    """Pseudo-test cross-validation for the Gaussian pillar on train sessions."""
    md = load_metadata(config)
    train_md = md.loc[md["split"] == "train", ["session_id", "day"]].copy().sort_values("day")
    train_md = train_md.reset_index(drop=True)
    train_md["fold"] = np.arange(len(train_md)) % max(n_folds, 1)

    records: list[dict[str, Any]] = []
    for _, row in tqdm(train_md.iterrows(), total=len(train_md), desc="Gaussian CV sessions"):
        sid = str(row["session_id"])
        sess = load_train_session(sid, config)
        sbp = np.asarray(sess["sbp"], dtype=np.float32)
        masked_sbp, mask = _pseudo_test_mask_from_train_session(
            sbp=sbp,
            trial_starts=np.asarray(sess["trial_starts"], dtype=np.int64),
            trial_ends=np.asarray(sess["trial_ends"], dtype=np.int64),
            n_masked_trials=min(10, int(sess["n_trials"])),
            n_channels_to_mask=config.masked_channels_per_bin,
            seed=config.seed + int(sid[1:]),
        )
        # Fit on fully observed rows only (pseudo-test protocol).
        fully_observed = ~mask.any(axis=1)
        if not fully_observed.any():
            raise ValueError(f"No fully observed rows available for pseudo-test Gaussian fit in {sid}")
        cg = ConditionalGaussian(solve_eps=config.gaussian_solve_eps).fit(sbp[fully_observed])
        pred_dense = cg.predict(masked_sbp, mask)
        sol = build_pseudo_solution_df(sid, sbp, mask)
        sub = build_submission_like_df_from_dense(mask, pred_dense)
        nmse = metric.score(sol, sub, row_id_column_name="sample_id")
        records.append({"session_id": sid, "day": int(row["day"]), "fold": int(row["fold"]), "nmse": float(nmse)})

    out = pd.DataFrame(records).sort_values(["fold", "session_id"]).reset_index(drop=True)
    out_path = config.results_dir / "gaussian_cv.csv"
    out.to_csv(out_path, index=False)
    LOGGER.info("Saved Gaussian CV results to %s", out_path)
    return out


@torch.no_grad()
def _predict_dense_mae_with_day(
    model,
    sbp_raw: np.ndarray,
    mask: np.ndarray,
    day: int,
    means: np.ndarray,
    stds: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    sbp_z = (sbp_raw - means[None, :]) / stds[None, :]
    sbp_z = sbp_z.astype(np.float32)
    sbp_z[mask] = 0.0
    pred_z = sbp_z.copy()
    rows = np.where(mask.any(axis=1))[0].astype(np.int64)
    for start in range(0, len(rows), batch_size):
        r = rows[start : start + batch_size]
        windows = _gather_windows_numpy(sbp_z, r, model.context_bins)
        row_masks = mask[r]
        for i in range(windows.shape[0]):
            windows[i, :, row_masks[i]] = 0.0
        x = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
        m = torch.from_numpy(row_masks).to(device=device, dtype=torch.bool)
        days = torch.full((x.shape[0],), float(day), device=device, dtype=torch.float32)
        out = model(sbp_window=x, mask=m, session_days=days)
        preds = out["pred_values"].cpu().numpy()
        idx = out["masked_channel_idx"].cpu().numpy()
        pad = out["masked_padding_mask"].cpu().numpy().astype(bool)
        for i, row_idx in enumerate(r.tolist()):
            valid = ~pad[i]
            pred_z[row_idx, idx[i, valid]] = preds[i, valid]
    return (pred_z * stds[None, :] + means[None, :]).astype(np.float32)



def cross_validate_mae(config: Config, checkpoint_path: Path, n_folds: int = 5) -> pd.DataFrame:
    """Pseudo-test MAE cross-validation on train sessions using a pretrained checkpoint.

    This evaluates the pretrained MAE (without per-session TTT on train pseudo-folds) by masking
    held-out trials within each train session and scoring reconstruction NMSE.
    """
    md = load_metadata(config)
    train_md = md.loc[md["split"] == "train", ["session_id", "day"]].copy().sort_values("day").reset_index(drop=True)
    train_md["fold"] = np.arange(len(train_md)) % max(n_folds, 1)

    device = torch.device(config.device)
    model, _, model_cfg = load_pretrained_model(config, checkpoint_path, device)
    model.eval()

    records: list[dict[str, Any]] = []
    for _, row in tqdm(train_md.iterrows(), total=len(train_md), desc="MAE CV sessions"):
        sid = str(row["session_id"])
        sess = load_train_session(sid, model_cfg)
        sbp = np.asarray(sess["sbp"], dtype=np.float32)
        masked_sbp, mask = _pseudo_test_mask_from_train_session(
            sbp=sbp,
            trial_starts=np.asarray(sess["trial_starts"], dtype=np.int64),
            trial_ends=np.asarray(sess["trial_ends"], dtype=np.int64),
            n_masked_trials=min(10, int(sess["n_trials"])),
            n_channels_to_mask=model_cfg.masked_channels_per_bin,
            seed=model_cfg.seed + 5000 + int(sid[1:]),
        )
        _, means, stds = zscore_session(sbp, return_params=True)
        pred_dense = _predict_dense_mae_with_day(
            model=model,
            sbp_raw=masked_sbp,
            mask=mask,
            day=int(row["day"]),
            means=means,
            stds=stds,
            batch_size=min(model_cfg.mae_batch_size, 512),
            device=device,
        )
        sol = build_pseudo_solution_df(sid, sbp, mask)
        sub = build_submission_like_df_from_dense(mask, pred_dense)
        nmse = metric.score(sol, sub, row_id_column_name="sample_id")
        records.append({"session_id": sid, "day": int(row["day"]), "fold": int(row["fold"]), "nmse": float(nmse)})

    out = pd.DataFrame(records).sort_values(["fold", "session_id"]).reset_index(drop=True)
    out_path = config.results_dir / "mae_cv.csv"
    out.to_csv(out_path, index=False)
    LOGGER.info("Saved MAE CV results to %s", out_path)
    return out



def analyze_by_difficulty(predictions_df: pd.DataFrame, solution_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Compute NMSE grouped by metadata difficulty for test sessions."""
    md = load_metadata(config)[["session_id", "difficulty"]].copy()
    merged = solution_df.merge(predictions_df[["sample_id", "predicted_sbp"]], on="sample_id", how="left")
    if merged["predicted_sbp"].isna().any():
        raise ValueError("Predictions missing sample IDs in analyze_by_difficulty")
    merged = merged.merge(md, on="session_id", how="left")

    rows: list[dict[str, Any]] = []
    for difficulty, group in merged.groupby("difficulty", dropna=False):
        score = metric.score(
            group[["sample_id", "true_sbp", "session_id", "channel", "channel_var"]].copy(),
            group[["sample_id", "predicted_sbp"]].copy(),
            row_id_column_name="sample_id",
        )
        rows.append({"difficulty": difficulty if pd.notna(difficulty) else "NA", "nmse": float(score), "n_rows": int(len(group))})
    out = pd.DataFrame(rows).sort_values("difficulty").reset_index(drop=True)
    out.to_csv(config.results_dir / "analysis_by_difficulty.csv", index=False)
    return out



def analyze_by_channel(predictions_df: pd.DataFrame, solution_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Compute NMSE per channel across all sessions."""
    merged = solution_df.merge(predictions_df[["sample_id", "predicted_sbp"]], on="sample_id", how="left")
    if merged["predicted_sbp"].isna().any():
        raise ValueError("Predictions missing sample IDs in analyze_by_channel")

    rows: list[dict[str, Any]] = []
    for channel, group in merged.groupby("channel"):
        score = metric.score(
            group[["sample_id", "true_sbp", "session_id", "channel", "channel_var"]].copy(),
            group[["sample_id", "predicted_sbp"]].copy(),
            row_id_column_name="sample_id",
        )
        rows.append({"channel": int(channel), "nmse": float(score), "n_rows": int(len(group))})
    out = pd.DataFrame(rows).sort_values("nmse", ascending=False).reset_index(drop=True)
    out.to_csv(config.results_dir / "analysis_by_channel.csv", index=False)
    return out



def plot_drift_vs_nmse(predictions_df: pd.DataFrame, solution_df: pd.DataFrame, config: Config) -> Path:
    """Plot per-session drift gap vs NMSE and save to `results/plots`."""
    md = load_metadata(config)[["session_id", "days_from_nearest_train"]].copy()
    merged = solution_df.merge(predictions_df[["sample_id", "predicted_sbp"]], on="sample_id", how="left")
    if merged["predicted_sbp"].isna().any():
        raise ValueError("Predictions missing sample IDs in plot_drift_vs_nmse")

    rows: list[dict[str, Any]] = []
    for session_id, group in merged.groupby("session_id"):
        nmse = metric.score(
            group[["sample_id", "true_sbp", "session_id", "channel", "channel_var"]].copy(),
            group[["sample_id", "predicted_sbp"]].copy(),
            row_id_column_name="sample_id",
        )
        rows.append({"session_id": session_id, "nmse": float(nmse)})
    summary = pd.DataFrame(rows).merge(md, on="session_id", how="left")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(summary["days_from_nearest_train"], summary["nmse"], alpha=0.8)
    ax.set_xlabel("Days from nearest train session")
    ax.set_ylabel("NMSE")
    ax.set_title("Drift vs Reconstruction NMSE")
    ax.grid(True, alpha=0.3)
    out_path = config.plots_dir / "drift_vs_nmse.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local evaluation utilities")
    parser.add_argument("--config", choices=["local", "hpc"], default="local")
    parser.add_argument("--gaussian-cv", action="store_true")
    parser.add_argument("--mae-cv", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--analysis-from-files", action="store_true")
    parser.add_argument("--predictions_csv", type=str, default=None)
    parser.add_argument("--solution_csv", type=str, default=None)
    return parser.parse_args()



def main() -> None:
    """CLI entry point for local evaluation and analysis."""
    args = _parse_args()
    config = get_config(args.config)
    ensure_output_dirs(config)
    setup_logging(config.log_level, config.logs_dir / "evaluate.log")
    preflight_validate_submission_indices(config)

    if args.gaussian_cv:
        cross_validate_gaussian(config, n_folds=args.n_folds)

    if args.mae_cv:
        if args.checkpoint is None:
            raise ValueError("--mae-cv requires --checkpoint")
        cross_validate_mae(config, checkpoint_path=Path(args.checkpoint), n_folds=args.n_folds)

    if args.analysis_from_files:
        if args.predictions_csv is None or args.solution_csv is None:
            raise ValueError("--analysis-from-files requires --predictions_csv and --solution_csv")
        pred_df = pd.read_csv(args.predictions_csv)
        sol_df = pd.read_csv(args.solution_csv)
        by_diff = analyze_by_difficulty(pred_df, sol_df, config)
        by_channel = analyze_by_channel(pred_df, sol_df, config)
        plot_path = plot_drift_vs_nmse(pred_df, sol_df, config)
        LOGGER.info("Saved analysis outputs: by_difficulty=%d rows, by_channel=%d rows, plot=%s", len(by_diff), len(by_channel), plot_path)


if __name__ == "__main__":
    main()
