"""Run test inference with best config from sweep on ctx=800 checkpoint.

Generates two submissions:
1. stride=ctx/8 + tta5_strong + sigma=0 (best val)
2. stride=ctx/8 + no TTA + sigma=0 (safe fallback)

Usage:
    modal run --detach phase2/modal/infer.py
"""

import modal

app = modal.App("phase2-infer-best")

data_vol = modal.Volume.from_name("phase2-data", create_if_missing=True)
output_vol = modal.Volume.from_name("phase2-outputs-gru-ctx800", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pandas", "scipy", "tqdm")
)

code_image = (
    base_image
    .add_local_file("phase2/config.py", "/root/repo/config.py")
    .add_local_file("phase2/data.py", "/root/repo/data.py")
    .add_local_file("phase2/model.py", "/root/repo/model.py")
    .add_local_file("phase2/train.py", "/root/repo/train.py")
    .add_local_file("phase2/inference.py", "/root/repo/inference.py")
)


@app.function(
    image=code_image,
    gpu="A10G",
    timeout=7200,
    volumes={"/root/data": data_vol, "/root/outputs": output_vol},
)
def run_best_inference():
    import sys
    import time
    import json
    import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch

    sys.path.insert(0, "/root/repo")

    from config import Phase2Config, set_global_seeds
    from data import (
        discover_session_ids, load_sample_submission,
        zscore_session, load_sbp, get_active_mask,
    )
    from model import build_model

    config = Phase2Config(
        profile="modal",
        repo_root=Path("/root/repo"),
        data_dir=Path("/root/data"),
        train_dir=Path("/root/data/train"),
        test_dir=Path("/root/data/test"),
        output_dir=Path("/root/outputs"),
        checkpoints_dir=Path("/root/outputs/checkpoints"),
        results_dir=Path("/root/outputs/results"),
        logs_dir=Path("/root/outputs/logs"),
        metadata_path=Path("/root/data/metadata.csv"),
        sample_sub_path=Path("/root/data/sample_submission.csv"),
        test_index_path=Path("/root/data/test_index.csv"),
        device="cuda",
        model_type="gru",
        context_bins=800,
        seed=44,
        gru_d_model=128,
        gru_n_layers=3,
        gru_dropout=0.2,
    )

    set_global_seeds(config.seed)
    data_vol.reload()
    output_vol.reload()

    device = torch.device("cuda")
    ctx = config.context_bins

    # Load model
    model = build_model(config).to(device)
    ckpt_path = config.checkpoints_dir / "best_gru.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_r2={ckpt.get('val_r2', '?')}")

    # Load test data
    test_ids = discover_session_ids(config.test_dir)
    sample_sub = load_sample_submission(config)
    print(f"Test sessions: {len(test_ids)}, submission rows: {len(sample_sub)}")

    def predict_session(sbp, active_mask, stride_div, tta_passes, noise_std, drop_frac):
        n_bins = sbp.shape[0]
        stride = max(1, ctx // stride_div)
        starts = list(range(0, n_bins - ctx + 1, stride))
        if starts and (starts[-1] + ctx) < n_bins:
            starts.append(n_bins - ctx)
        if not starts:
            return np.full((n_bins, 2), 0.5, dtype=np.float32)

        mask_t = torch.from_numpy(active_mask).to(device)
        rng = np.random.RandomState(42)

        all_preds = np.zeros((n_bins, 2), dtype=np.float64)
        all_counts = np.zeros(n_bins, dtype=np.float64)

        for tta_i in range(tta_passes):
            preds = np.zeros((n_bins, 2), dtype=np.float64)
            counts = np.zeros(n_bins, dtype=np.float64)

            for batch_start in range(0, len(starts), 32):
                batch_indices = starts[batch_start:batch_start + 32]
                batch_sbp = []
                for s in batch_indices:
                    chunk = sbp[s:s + ctx].copy()
                    if tta_i > 0:
                        if noise_std > 0:
                            noise = rng.randn(*chunk.shape).astype(np.float32) * noise_std
                            noise[:, ~active_mask] = 0
                            chunk = chunk + noise
                        if drop_frac > 0:
                            n_active = active_mask.sum()
                            n_drop = int(n_active * drop_frac * rng.random())
                            if n_drop > 0:
                                active_idx = np.where(active_mask)[0]
                                drop_idx = rng.choice(active_idx, n_drop, replace=False)
                                chunk[:, drop_idx] = 0
                    batch_sbp.append(chunk)

                sbp_t = torch.from_numpy(np.stack(batch_sbp)).to(device)
                mask_batch = mask_t.unsqueeze(0).expand(len(batch_indices), -1)

                with torch.no_grad():
                    out = model(sbp_t, active_mask=mask_batch)
                out_np = out.cpu().numpy()

                for i, s in enumerate(batch_indices):
                    preds[s:s + ctx] += out_np[i]
                    counts[s:s + ctx] += 1.0

            counts = np.maximum(counts, 1.0)
            preds /= counts[:, None]
            all_preds += preds
            all_counts += 1.0

        all_preds /= all_counts[:, None]
        return np.clip(all_preds, 0.0, 1.0).astype(np.float32)

    def build_submission(all_predictions, tag):
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
        sub["index_pos"] = sub["index_pos"].where(np.isfinite(sub["index_pos"]), 0.5)
        sub["mrp_pos"] = sub["mrp_pos"].where(np.isfinite(sub["mrp_pos"]), 0.5)

        fname = f"submission_gru_ctx800_seed44_raw_{tag}.csv"
        out_path = config.results_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)
        print(f"Saved: {fname} ({len(sub)} rows)")

    configs = [
        ("stride8_tta5strong", 8, 5, 0.1, 0.2),
        ("stride8_noTTA", 8, 1, 0.0, 0.0),
    ]

    for tag, stride_div, tta_passes, noise_std, drop_frac in configs:
        print(f"\n--- {tag} ---")
        t0 = time.time()
        all_predictions = {}
        for i, sid in enumerate(test_ids):
            raw_sbp = load_sbp(config.test_dir, sid)
            z_sbp, _, _ = zscore_session(raw_sbp)
            active = get_active_mask(raw_sbp)
            all_predictions[sid] = predict_session(
                z_sbp, active, stride_div, tta_passes, noise_std, drop_frac
            )
            if (i + 1) % 25 == 0 or (i + 1) == len(test_ids):
                print(f"  Predicted {i+1}/{len(test_ids)} sessions ({time.time()-t0:.1f}s)")

        build_submission(all_predictions, tag)

    output_vol.commit()
    print("\nDone! Download with:")
    print("  modal volume get phase2-outputs-gru-ctx800 results/ .")


@app.local_entrypoint()
def main():
    run_best_inference.remote()
