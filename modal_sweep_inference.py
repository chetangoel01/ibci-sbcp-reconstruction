"""Sweep inference strategies on existing ctx=800 GRU checkpoint.

Tests combinations of:
- Stride: ctx//2, ctx//4, ctx//8
- TTA: off, 3 passes, 5 passes (with noise + channel dropout)
- Smoothing: sigma 0, 3

Evaluates on val set. No retraining.

Usage:
    modal run --detach modal_sweep_inference.py
"""

import modal

app = modal.App("phase2-sweep-inference")

data_vol = modal.Volume.from_name("phase2-data", create_if_missing=True)
output_vol = modal.Volume.from_name("phase2-outputs-gru-ctx800", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pandas", "scipy", "tqdm")
)

code_image = (
    base_image
    .add_local_file("phase2_config.py", "/root/repo/phase2_config.py")
    .add_local_file("phase2_data.py", "/root/repo/phase2_data.py")
    .add_local_file("phase2_model.py", "/root/repo/phase2_model.py")
    .add_local_file("phase2_train.py", "/root/repo/phase2_train.py")
    .add_local_file("phase2_inference.py", "/root/repo/phase2_inference.py")
)


@app.function(
    image=code_image,
    gpu="A10G",
    timeout=7200,
    volumes={"/root/data": data_vol, "/root/outputs": output_vol},
)
def sweep_inference():
    import sys
    from pathlib import Path

    import numpy as np
    import torch
    from scipy.ndimage import gaussian_filter1d

    sys.path.insert(0, "/root/repo")

    from phase2_config import Phase2Config, set_global_seeds
    from phase2_data import SessionCache, discover_session_ids, split_train_val
    from phase2_model import build_model
    from phase2_train import compute_r2

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

    # Load model
    model = build_model(config).to(device)
    ckpt_path = config.checkpoints_dir / "best_gru.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_r2={ckpt.get('val_r2', '?')}")

    # Load val data
    all_train_ids = discover_session_ids(config.train_dir)
    _, val_ids = split_train_val(all_train_ids, n_val=config.val_sessions, seed=config.seed)
    val_cache = SessionCache(config.train_dir, val_ids, has_kinematics=True)
    print(f"Validation sessions: {len(val_ids)}")

    ctx = config.context_bins

    def predict_session_custom(sbp, active_mask, stride_div, tta_passes, noise_std, drop_frac):
        """Predict a session with custom stride and TTA."""
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
                        # Add noise
                        if noise_std > 0:
                            noise = rng.randn(*chunk.shape).astype(np.float32) * noise_std
                            noise[:, ~active_mask] = 0
                            chunk = chunk + noise
                        # Random channel dropout
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
        return all_preds.astype(np.float32)

    # Define sweep
    stride_divs = [2, 4, 8]
    tta_configs = [
        ("no_tta", 1, 0.0, 0.0),
        ("tta3", 3, 0.05, 0.1),
        ("tta5", 5, 0.05, 0.1),
        ("tta5_strong", 5, 0.1, 0.2),
    ]
    sigmas = [0, 3]

    results = []
    print(f"\n{'stride':>10} | {'tta':>12} | {'sigma':>5} | {'val_R2':>8}")
    print("-" * 50)

    for stride_div in stride_divs:
        for tta_name, tta_passes, noise_std, drop_frac in tta_configs:
            # Generate raw predictions for all val sessions
            session_preds = {}
            for sid in val_ids:
                sbp_np = val_cache.sbp[sid]
                active_np = val_cache.active_masks[sid]
                session_preds[sid] = predict_session_custom(
                    sbp_np, active_np, stride_div, tta_passes, noise_std, drop_frac
                )

            for sigma in sigmas:
                all_r2 = []
                for sid in val_ids:
                    pred = session_preds[sid].copy()
                    true = val_cache.kinematics[sid][:, :2]

                    if sigma > 0:
                        for c in range(2):
                            pred[:, c] = gaussian_filter1d(pred[:, c], sigma=sigma)

                    for c in range(2):
                        r2 = compute_r2(pred[:, c], true[:, c])
                        all_r2.append(r2)

                mean_r2 = float(np.mean(all_r2))
                results.append({
                    "stride_div": stride_div,
                    "tta": tta_name,
                    "sigma": sigma,
                    "val_r2": mean_r2,
                })
                print(f"  ctx/{stride_div:>3} | {tta_name:>12} | {sigma:>5} | {mean_r2:>8.4f}")

    # Sort by val R2
    results.sort(key=lambda x: x["val_r2"], reverse=True)
    print(f"\n=== TOP 5 CONFIGS ===")
    for r in results[:5]:
        print(f"  stride=ctx/{r['stride_div']}, {r['tta']}, sigma={r['sigma']} -> val_R2={r['val_r2']:.4f}")

    best = results[0]
    print(f"\nBest: stride=ctx/{best['stride_div']}, {best['tta']}, sigma={best['sigma']} -> val_R2={best['val_r2']:.4f}")

    return results


@app.local_entrypoint()
def main():
    results = sweep_inference.remote()
    print("\n=== FULL RESULTS ===")
    for r in sorted(results, key=lambda x: -x["val_r2"]):
        print(f"  stride=ctx/{r['stride_div']}, {r['tta']:>12}, sigma={r['sigma']} -> val_R2={r['val_r2']:.4f}")
