"""Sweep smoothing sigma on existing GRU checkpoint (no retraining).

Evaluates val R² for each sigma value, then generates submission CSV
for the best sigma. Fast — inference only, ~5 min total.

Usage:
    modal run modal_sweep_smoothing.py
"""

import modal

app = modal.App("phase2-smooth-sweep")

data_vol = modal.Volume.from_name("phase2-data", create_if_missing=True)
output_vol = modal.Volume.from_name("phase2-outputs-gru", create_if_missing=True)

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
    timeout=3600,
    volumes={"/root/data": data_vol, "/root/outputs": output_vol},
)
def sweep_smoothing():
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

    SIGMAS = [0, 1, 2, 3, 5, 8, 10, 12, 15, 20]

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
        context_bins=200,
        gru_d_model=128,
        gru_n_layers=3,
        gru_dropout=0.2,
        seed=44,
    )

    set_global_seeds(config.seed)
    data_vol.reload()
    output_vol.reload()

    device = torch.device("cuda")

    # --- Load model ---
    model = build_model(config).to(device)
    ckpt_path = config.checkpoints_dir / "best_gru.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_r2={ckpt.get('val_r2', '?')}")

    # --- Load val data ---
    all_train_ids = discover_session_ids(config.train_dir)
    _, val_ids = split_train_val(all_train_ids, n_val=config.val_sessions, seed=config.seed)
    print(f"Validation sessions: {len(val_ids)}")
    val_cache = SessionCache(config.train_dir, val_ids, has_kinematics=True)

    # --- Generate raw predictions for all val sessions (once) ---
    ctx = config.context_bins
    raw_preds: dict[str, np.ndarray] = {}
    true_pos: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for sid in val_ids:
            sbp_np = val_cache.sbp[sid]
            kin_np = val_cache.kinematics[sid]
            active_np = val_cache.active_masks[sid]
            n_bins = sbp_np.shape[0]

            preds = np.zeros((n_bins, 2), dtype=np.float64)
            counts = np.zeros(n_bins, dtype=np.float64)

            stride = max(1, ctx // 2)
            starts = list(range(0, n_bins - ctx + 1, stride))
            if starts and (starts[-1] + ctx) < n_bins:
                starts.append(n_bins - ctx)

            mask_t = torch.from_numpy(active_np).unsqueeze(0).to(device)

            for start in starts:
                sbp_t = torch.from_numpy(sbp_np[start : start + ctx]).unsqueeze(0).to(device)
                out = model(sbp_t, active_mask=mask_t)
                out_np = out.squeeze(0).cpu().numpy()
                preds[start : start + ctx] += out_np
                counts[start : start + ctx] += 1.0

            counts = np.maximum(counts, 1.0)
            preds /= counts[:, None]

            raw_preds[sid] = preds.astype(np.float32)
            true_pos[sid] = kin_np[:, :2]

    print("Raw predictions generated for all val sessions.")

    # --- Sweep sigma ---
    print(f"\n{'sigma':>8} | {'val_R2':>8}")
    print("-" * 22)

    results = []
    for sigma in SIGMAS:
        all_r2 = []
        for sid in val_ids:
            pred = raw_preds[sid].copy()
            if sigma > 0:
                for c in range(2):
                    pred[:, c] = gaussian_filter1d(pred[:, c], sigma=sigma)

            true = true_pos[sid]
            for c in range(2):
                r2 = compute_r2(pred[:, c], true[:, c])
                all_r2.append(r2)

        mean_r2 = float(np.mean(all_r2))
        results.append((sigma, mean_r2))
        marker = " <-- best so far" if not any(r[1] > mean_r2 for r in results[:-1]) else ""
        print(f"{sigma:>8.1f} | {mean_r2:>8.4f}{marker}")

    best_sigma, best_r2 = max(results, key=lambda x: x[1])
    print(f"\nBest sigma: {best_sigma} (val R2 = {best_r2:.4f})")

    # --- Generate submission with best sigma ---
    print(f"\nRunning test inference with sigma={best_sigma}...")

    from phase2_inference import run_inference as infer_fn

    infer_fn(config, smooth_sigma=float(best_sigma))
    output_vol.commit()

    sigma_str = f"sigma{best_sigma:g}" if best_sigma > 0 else "raw"
    fname = f"submission_gru_ctx200_seed44_{sigma_str}.csv"
    print(f"\nDone! Submission saved as {fname}")
    print("Download with:")
    print(f"  modal volume get phase2-outputs-gru results/{fname} .")

    return {"results": results, "best_sigma": best_sigma, "best_val_r2": best_r2}


@app.local_entrypoint()
def main():
    result = sweep_smoothing.remote()
    print("\n=== SMOOTHING SWEEP RESULTS ===")
    for sigma, r2 in result["results"]:
        print(f"  sigma={sigma:>5.1f}  val_R2={r2:.4f}")
    print(f"\nBest: sigma={result['best_sigma']} -> val_R2={result['best_val_r2']:.4f}")
