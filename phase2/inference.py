"""Inference script for Phase 2: generate submission.csv from trained model.

Usage:
    python phase2_inference.py --model transformer
    python phase2_inference.py --model poyo --checkpoint phase2_outputs/checkpoints/best_poyo.pt
    python phase2_inference.py --model transformer --smooth_sigma 2.0
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn

from config import Phase2Config, get_config, set_global_seeds
from data import (
    SessionCache,
    discover_session_ids,
    load_sample_submission,
    zscore_session,
    load_sbp,
    get_active_mask,
)
from model import build_model

LOGGER = logging.getLogger("phase2_inference")


def predict_session(
    model: nn.Module,
    sbp: np.ndarray,
    active_mask: np.ndarray,
    context_bins: int,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate predictions for a full session by tiling overlapping windows.

    Returns:
        (n_bins, 2) predicted positions
    """
    n_bins = sbp.shape[0]
    preds = np.zeros((n_bins, 2), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    stride = max(1, context_bins // 2)
    starts = list(range(0, n_bins - context_bins + 1, stride))
    if starts and (starts[-1] + context_bins) < n_bins:
        starts.append(n_bins - context_bins)

    # Handle edge case: session shorter than context window
    if not starts:
        pad_len = context_bins - n_bins
        padded_sbp = np.pad(sbp, ((0, pad_len), (0, 0)), mode="edge")
        sbp_t = torch.from_numpy(padded_sbp).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(active_mask).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(sbp_t, active_mask=mask_t)
        return out.squeeze(0).cpu().numpy()[:n_bins].astype(np.float32)

    mask_t = torch.from_numpy(active_mask).to(device)

    for batch_start in range(0, len(starts), batch_size):
        batch_indices = starts[batch_start : batch_start + batch_size]
        batch_sbp = []
        for s in batch_indices:
            batch_sbp.append(sbp[s : s + context_bins])

        sbp_t = torch.from_numpy(np.stack(batch_sbp)).to(device)
        mask_batch = mask_t.unsqueeze(0).expand(len(batch_indices), -1)

        with torch.no_grad():
            out = model(sbp_t, active_mask=mask_batch)

        out_np = out.cpu().numpy()

        for i, s in enumerate(batch_indices):
            e = s + context_bins
            preds[s:e] += out_np[i]
            counts[s:e] += 1.0

    counts = np.maximum(counts, 1.0)
    preds /= counts[:, None]
    return preds.astype(np.float32)


def run_inference(config: Phase2Config, checkpoint_path: Path | None = None, smooth_sigma: float = 0.0, tag: str | None = None) -> None:
    set_global_seeds(config.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    device = torch.device(config.device)
    LOGGER.info("Device: %s | Model: %s", device, config.model_type)

    # --- Load model ---
    model = build_model(config).to(device)

    ckpt_path = checkpoint_path or (config.checkpoints_dir / f"best_{config.model_type}.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    LOGGER.info("Loaded checkpoint: %s (epoch %d, val_r2=%.4f)", ckpt_path, ckpt["epoch"], ckpt.get("val_r2", 0))

    # --- Load test data ---
    test_ids = discover_session_ids(config.test_dir)
    LOGGER.info("Test sessions: %d", len(test_ids))

    # --- Load sample submission for alignment ---
    sample_sub = load_sample_submission(config)
    LOGGER.info("Sample submission rows: %d", len(sample_sub))

    # --- Generate predictions per session ---
    all_predictions: dict[str, np.ndarray] = {}
    t0 = time.time()

    for i, sid in enumerate(test_ids):
        raw_sbp = load_sbp(config.test_dir, sid)
        z_sbp, _, _ = zscore_session(raw_sbp)
        active = get_active_mask(raw_sbp)

        preds = predict_session(model, z_sbp, active, config.context_bins, device)

        # Clip to valid position range
        preds = np.clip(preds, 0.0, 1.0)

        # Optional temporal smoothing
        if smooth_sigma > 0:
            for c in range(2):
                preds[:, c] = gaussian_filter1d(preds[:, c], sigma=smooth_sigma)
                preds[:, c] = np.clip(preds[:, c], 0.0, 1.0)

        all_predictions[sid] = preds

        if (i + 1) % 25 == 0 or (i + 1) == len(test_ids):
            LOGGER.info("  Predicted %d/%d sessions (%.1fs)", i + 1, len(test_ids), time.time() - t0)

    # --- Build submission by vectorized fill ---
    LOGGER.info("Building submission CSV...")
    sub = sample_sub.copy()
    index_pos_arr = sub["index_pos"].values.astype(np.float64)
    mrp_pos_arr = sub["mrp_pos"].values.astype(np.float64)

    for sid in test_ids:
        if sid not in all_predictions:
            continue
        pred = all_predictions[sid]
        mask = (sub["session_id"] == sid).values
        time_bins = sub.loc[mask, "time_bin"].values
        valid = time_bins < len(pred)
        row_indices = np.where(mask)[0][valid]
        tb = time_bins[valid]
        index_pos_arr[row_indices] = pred[tb, 0]
        mrp_pos_arr[row_indices] = pred[tb, 1]

    sub["index_pos"] = index_pos_arr
    sub["mrp_pos"] = mrp_pos_arr

    # Guard against NaN
    sub["index_pos"] = sub["index_pos"].where(np.isfinite(sub["index_pos"]), 0.5)
    sub["mrp_pos"] = sub["mrp_pos"].where(np.isfinite(sub["mrp_pos"]), 0.5)

    # Build descriptive filename
    sigma_str = f"sigma{smooth_sigma:g}" if smooth_sigma > 0 else "raw"
    fname = f"submission_{config.model_type}_ctx{config.context_bins}_seed{config.seed}_{sigma_str}"
    if tag:
        fname += f"_{tag}"
    fname += ".csv"

    out_path = config.results_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    LOGGER.info("Submission saved: %s (%d rows)", out_path, len(sub))

    root_path = config.repo_root / fname
    sub.to_csv(root_path, index=False)
    LOGGER.info("Also saved: %s", root_path)

    import json, datetime
    meta = {
        "model_type": config.model_type,
        "checkpoint": str(checkpoint_path) if checkpoint_path else "best",
        "checkpoint_epoch": ckpt.get("epoch", "unknown"),
        "checkpoint_val_r2": ckpt.get("val_r2", "unknown"),
        "smooth_sigma": smooth_sigma,
        "context_bins": config.context_bins,
        "n_test_sessions": len(test_ids),
        "n_rows": len(sub),
        "generated_at": datetime.datetime.now().isoformat(),
    }
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    LOGGER.info("Metadata saved: %s", meta_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 inference")
    parser.add_argument("--model", choices=["transformer", "poyo", "gru"], default="transformer")
    parser.add_argument("--profile", choices=["local", "hpc"], default="local")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--smooth_sigma", type=float, default=0.0, help="Gaussian smoothing sigma (0=off)")
    parser.add_argument("--context_bins", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Optional tag appended to submission filename")
    args = parser.parse_args()

    config = get_config(args.profile)
    overrides: dict = {"model_type": args.model}
    if args.context_bins is not None:
        overrides["context_bins"] = args.context_bins
    config = replace(config, **overrides)

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    run_inference(config, checkpoint_path=ckpt, smooth_sigma=args.smooth_sigma, tag=args.tag)


if __name__ == "__main__":
    main()
