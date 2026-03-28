"""Generate submission CSVs for models that need them, then create ensembles."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

from phase2_config import get_config, set_global_seeds
from phase2_data import (
    discover_session_ids, load_sbp, zscore_session, get_active_mask,
    load_sample_submission,
)
from phase2_model import GRUDecoder


def predict_session(model, sbp, active, ctx, device):
    n_bins = sbp.shape[0]
    preds = np.zeros((n_bins, 2), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    stride = max(1, ctx // 2)
    starts = list(range(0, n_bins - ctx + 1, stride))
    if starts and (starts[-1] + ctx) < n_bins:
        starts.append(n_bins - ctx)

    if not starts:
        pad_len = ctx - n_bins
        padded = np.pad(sbp, ((0, pad_len), (0, 0)), mode="edge")
        inp = torch.from_numpy(padded).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
        return out.squeeze(0).cpu().numpy()[:n_bins].astype(np.float32)

    for batch_start in range(0, len(starts), 32):
        batch_idx = starts[batch_start:batch_start + 32]
        batch_inp = np.stack([sbp[s:s + ctx] for s in batch_idx])
        inp = torch.from_numpy(batch_inp).to(device)
        with torch.no_grad():
            out = model(inp)
        out_np = out.cpu().numpy()
        for j, s in enumerate(batch_idx):
            preds[s:s + ctx] += out_np[j]
            counts[s:s + ctx] += 1.0

    counts = np.maximum(counts, 1.0)
    preds /= counts[:, None]
    return preds.astype(np.float32)


def generate_submission(ckpt_path, config, device, sigma=3.0):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    ctx = cfg["context_bins"]

    model = GRUDecoder(
        n_channels=96, d_model=cfg["gru_d_model"],
        n_layers=cfg["gru_n_layers"], dropout=cfg["gru_dropout"], n_outputs=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    print(f"  Loaded: d={cfg['gru_d_model']} ctx={ctx} val_r2={ckpt.get('val_r2', '?'):.4f}")

    test_ids = discover_session_ids(config.test_dir)
    sample_sub = load_sample_submission(config)
    all_preds = {}

    t0 = time.time()
    for i, sid in enumerate(test_ids):
        raw_sbp = load_sbp(config.test_dir, sid)
        z_sbp, _, _ = zscore_session(raw_sbp)
        active = get_active_mask(raw_sbp)
        pred = predict_session(model, z_sbp, active, ctx, device)
        pred = np.clip(pred, 0.0, 1.0)
        if sigma > 0:
            for c in range(2):
                pred[:, c] = np.clip(gaussian_filter1d(pred[:, c], sigma=sigma), 0.0, 1.0)
        all_preds[sid] = pred
        if (i + 1) % 25 == 0 or (i + 1) == len(test_ids):
            print(f"  {i+1}/{len(test_ids)} ({time.time()-t0:.1f}s)")

    # Fill submission
    sub = sample_sub.copy()
    idx_arr = sub["index_pos"].values.astype(np.float64)
    mrp_arr = sub["mrp_pos"].values.astype(np.float64)
    for sid in test_ids:
        pred = all_preds[sid]
        mask = (sub["session_id"] == sid).values
        tb = sub.loc[mask, "time_bin"].values
        valid = tb < len(pred)
        rows = np.where(mask)[0][valid]
        idx_arr[rows] = pred[tb[valid], 0]
        mrp_arr[rows] = pred[tb[valid], 1]
    sub["index_pos"] = idx_arr
    sub["mrp_pos"] = mrp_arr
    sub["index_pos"] = sub["index_pos"].where(np.isfinite(sub["index_pos"]), 0.5)
    sub["mrp_pos"] = sub["mrp_pos"].where(np.isfinite(sub["mrp_pos"]), 0.5)

    del model
    return sub


def main():
    set_global_seeds(42)
    config = get_config("local")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Generate gru_wide submission (doesn't exist yet)
    print("Generating gru_wide submission...")
    sub_wide = generate_submission(
        "phase2_outputs/checkpoints/gru_wide/best_gru.pt", config, device, sigma=3.0
    )
    sub_wide.to_csv("submission_gru_wide_d256_ctx400_sigma3.csv", index=False)
    print(f"Saved: submission_gru_wide_d256_ctx400_sigma3.csv\n")

    # Load existing submissions
    subs = {
        "gru_wide": sub_wide,
        "gru_ctx800": pd.read_csv("submission_gru_ctx800_seed44_sigma3.csv"),
        "gru_ctx600": pd.read_csv("submission_gru_ctx600_seed44_sigma3.csv"),
        "gru_ctx400": pd.read_csv("submission_gru_ctx400_seed44_sigma3.csv"),
        "gru_ctx200": pd.read_csv("submission_gru_ctx200_seed44_sigma3.csv"),
    }

    # Create ensembles with optimal weights from validation
    ensembles = [
        ("opt_top3", {"gru_wide": 0.382, "gru_ctx800": 0.372, "gru_ctx600": 0.247}),
        ("opt_top4", {"gru_wide": 0.380, "gru_ctx800": 0.370, "gru_ctx600": 0.245, "gru_ctx400": 0.005}),
        ("equal_top3", {"gru_wide": 0.333, "gru_ctx800": 0.333, "gru_ctx600": 0.334}),
        ("equal_2", {"gru_wide": 0.5, "gru_ctx800": 0.5}),
        ("equal_top4", {"gru_wide": 0.25, "gru_ctx800": 0.25, "gru_ctx600": 0.25, "gru_ctx400": 0.25}),
        ("equal_all5", {"gru_wide": 0.2, "gru_ctx800": 0.2, "gru_ctx600": 0.2, "gru_ctx400": 0.2, "gru_ctx200": 0.2}),
    ]

    ref = subs["gru_wide"]
    for ens_name, weights in ensembles:
        total_w = sum(weights.values())
        sub = ref.copy()
        for col in ["index_pos", "mrp_pos"]:
            blended = sum(
                (w / total_w) * subs[name][col].values
                for name, w in weights.items()
            )
            sub[col] = np.clip(blended, 0.0, 1.0)
        fname = f"submission_ensemble_{ens_name}.csv"
        sub.to_csv(fname, index=False)
        w_str = ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
        print(f"Saved: {fname}  [{w_str}]")

    print("\nDone! Files ready for submission.")


if __name__ == "__main__":
    main()
