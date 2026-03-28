"""Train GRU on ALL 312 sessions (train + test with LINK kinematics).

With kinematics for all sessions, the model sees the full temporal range
instead of just the first 60%. This should dramatically improve generalization.

Usage:
    modal run --detach modal_train_full312.py
"""

import modal

app = modal.App("phase2-gru-full312")

data_vol = modal.Volume.from_name("phase2-data", create_if_missing=True)
output_vol = modal.Volume.from_name("phase2-outputs-full312", create_if_missing=True)

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
    timeout=14400,
    volumes={"/root/data": data_vol, "/root/outputs": output_vol},
)
def train_full312():
    import sys
    import json
    import logging
    import math
    import time as time_mod
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from scipy.ndimage import gaussian_filter1d

    sys.path.insert(0, "/root/repo")

    from phase2_config import Phase2Config, ensure_output_dirs, set_global_seeds
    from phase2_data import (
        SessionCache, SlidingWindowDataset,
        discover_session_ids, load_sbp, load_sample_submission,
        zscore_session, get_active_mask, load_kinematics,
    )
    from phase2_model import GRUDecoder
    from phase2_train import compute_r2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    LOGGER = logging.getLogger("full312")

    data_vol.reload()

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
        batch_size=16,
        lr=3e-4,
        epochs=80,
        warmup_epochs=5,
        val_sessions=15,
        num_workers=4,
        velocity_aux_weight=0.1,
        seed=44,
        gru_d_model=128,
        gru_n_layers=3,
        gru_dropout=0.2,
    )

    set_global_seeds(config.seed)
    ensure_output_dirs(config)
    device = torch.device("cuda")

    # =========================================================================
    # Load ALL sessions (train + test)
    # =========================================================================
    train_ids = sorted(discover_session_ids(config.train_dir))
    test_ids = sorted(discover_session_ids(config.test_dir))

    # Check which test sessions have kinematics (from LINK)
    test_with_kin = [sid for sid in test_ids
                     if (config.test_dir / f"{sid}_kinematics.npy").exists()]
    LOGGER.info("Original train: %d, Test with kinematics: %d/%d",
                len(train_ids), len(test_with_kin), len(test_ids))

    # Combine all sessions with kinematics
    all_ids = sorted(train_ids + test_with_kin)
    LOGGER.info("Total sessions for training: %d", len(all_ids))

    # Split: use 15 random sessions as val, train on the rest
    # Include sessions from both original train and test for good coverage
    rng = np.random.RandomState(config.seed)
    shuffled = list(all_ids)
    rng.shuffle(shuffled)
    val_ids = sorted(shuffled[:config.val_sessions])
    full_train_ids = sorted(shuffled[config.val_sessions:])
    LOGGER.info("Full train: %d, Val: %d", len(full_train_ids), len(val_ids))

    # Load all sessions into cache
    # For test sessions, we need to load from test_dir
    LOGGER.info("Loading all sessions...")
    t0 = time_mod.time()

    # Build a unified cache manually since sessions come from different dirs
    class UnifiedCache:
        def __init__(self):
            self.sbp = {}
            self.kinematics = {}
            self.active_masks = {}
            self.stats = {}

    cache = UnifiedCache()

    for sid in all_ids:
        # Determine which directory
        if (config.train_dir / f"{sid}_sbp.npy").exists():
            data_dir = config.train_dir
        else:
            data_dir = config.test_dir

        raw_sbp = load_sbp(data_dir, sid)
        kin = load_kinematics(data_dir, sid)

        # Trim to matching length (LINK kinematics may differ slightly)
        min_len = min(raw_sbp.shape[0], kin.shape[0])
        raw_sbp = raw_sbp[:min_len]
        kin = kin[:min_len]

        active = get_active_mask(raw_sbp)

        # Z-score
        mean = np.zeros(96, dtype=np.float32)
        std = np.ones(96, dtype=np.float32)
        mean[active] = raw_sbp[:, active].mean(axis=0)
        std[active] = raw_sbp[:, active].std(axis=0)
        std[std < 1e-8] = 1.0
        z_sbp = np.zeros_like(raw_sbp)
        z_sbp[:, active] = (raw_sbp[:, active] - mean[active]) / std[active]

        cache.sbp[sid] = z_sbp
        cache.active_masks[sid] = active
        cache.stats[sid] = (mean, std)
        cache.kinematics[sid] = kin

    LOGGER.info("Loaded %d sessions (%.1fs)", len(all_ids), time_mod.time() - t0)

    # =========================================================================
    # Dataset + DataLoader
    # =========================================================================
    train_ds = SlidingWindowDataset(
        cache, full_train_ids,
        context_bins=config.context_bins,
        stride=max(1, config.context_bins // 2),
        samples_per_epoch=len(full_train_ids) * 200,
        augment=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # =========================================================================
    # Model
    # =========================================================================
    model = GRUDecoder(
        n_channels=config.n_channels,
        d_model=config.gru_d_model,
        n_layers=config.gru_n_layers,
        dropout=config.gru_dropout,
        n_outputs=config.n_position_outputs,
    ).to(device)
    LOGGER.info("GRU model: %d params", model.count_parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
        return config.lr_min_factor + 0.5 * (1 - config.lr_min_factor) * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # =========================================================================
    # Validation
    # =========================================================================
    def validate():
        model.train(False)
        ctx = config.context_bins
        all_r2 = []

        with torch.no_grad():
            for sid in val_ids:
                sbp_np = cache.sbp[sid]
                kin_np = cache.kinematics[sid]
                n_bins = sbp_np.shape[0]

                preds = np.zeros((n_bins, 2), dtype=np.float32)
                counts = np.zeros(n_bins, dtype=np.float32)

                stride = max(1, ctx // 2)
                starts = list(range(0, n_bins - ctx + 1, stride))
                if starts and (starts[-1] + ctx) < n_bins:
                    starts.append(n_bins - ctx)

                for start in starts:
                    inp = torch.from_numpy(sbp_np[start:start + ctx]).unsqueeze(0).to(device)
                    out = model(inp)
                    preds[start:start + ctx] += out.squeeze(0).cpu().numpy()
                    counts[start:start + ctx] += 1.0

                counts = np.maximum(counts, 1.0)
                preds /= counts[:, None]

                for c in range(2):
                    all_r2.append(compute_r2(preds[:, c], kin_np[:, c]))

        return float(np.mean(all_r2)) if all_r2 else 0.0

    # =========================================================================
    # Training loop
    # =========================================================================
    best_val_r2 = -float("inf")
    history = []

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time_mod.time()

        for batch in train_loader:
            sbp = batch["sbp"].to(device)
            positions = batch["positions"].to(device)
            velocities = batch["velocities"].to(device)

            pred = model(sbp)
            loss = nn.functional.mse_loss(pred, positions)

            if config.velocity_aux_weight > 0:
                vel_pred = pred[:, 1:, :] - pred[:, :-1, :]
                vel_true = velocities[:, :-1, :]
                loss = loss + config.velocity_aux_weight * nn.functional.mse_loss(vel_pred, vel_true)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        val_r2 = validate()
        is_best = val_r2 > best_val_r2
        if is_best:
            best_val_r2 = val_r2
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_r2": val_r2, "config": config.as_dict()},
                config.checkpoints_dir / "best_gru.pt",
            )

        LOGGER.info(
            "Epoch %3d/%d | loss=%.6f | val_R2=%.4f %s | lr=%.2e | %.1fs",
            epoch + 1, config.epochs, avg_loss, val_r2,
            " *" if is_best else "", optimizer.param_groups[0]["lr"],
            time_mod.time() - t0,
        )
        history.append({
            "epoch": epoch + 1, "train_loss": avg_loss,
            "val_r2": val_r2, "best_val_r2": best_val_r2,
        })

        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.epochs:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_r2": val_r2, "config": config.as_dict()},
                config.checkpoints_dir / "latest_gru.pt",
            )
            output_vol.commit()

    with open(config.results_dir / "train_history_full312.json", "w") as f:
        json.dump(history, f, indent=2)
    LOGGER.info("Training complete. Best val R2: %.4f", best_val_r2)

    # =========================================================================
    # Inference on test set
    # =========================================================================
    LOGGER.info("Loading best checkpoint for inference...")
    ckpt = torch.load(config.checkpoints_dir / "best_gru.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)

    sample_sub = load_sample_submission(config)
    ctx = config.context_bins

    for sigma in [0, 3]:
        LOGGER.info("Inference with sigma=%d...", sigma)
        t0 = time_mod.time()
        all_predictions = {}

        for i, sid in enumerate(test_ids):
            raw_sbp = load_sbp(config.test_dir, sid)
            z_sbp, _, _ = zscore_session(raw_sbp)
            active = get_active_mask(raw_sbp)
            n_bins = z_sbp.shape[0]

            preds = np.zeros((n_bins, 2), dtype=np.float64)
            counts = np.zeros(n_bins, dtype=np.float64)

            stride = max(1, ctx // 2)
            starts = list(range(0, n_bins - ctx + 1, stride))
            if starts and (starts[-1] + ctx) < n_bins:
                starts.append(n_bins - ctx)

            if not starts:
                pad_len = ctx - n_bins
                padded = np.pad(z_sbp, ((0, pad_len), (0, 0)), mode="edge")
                inp = torch.from_numpy(padded).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp)
                all_predictions[sid] = out.squeeze(0).cpu().numpy()[:n_bins].astype(np.float32)
                continue

            for batch_start in range(0, len(starts), 32):
                batch_idx = starts[batch_start:batch_start + 32]
                batch_inp = np.stack([z_sbp[s:s + ctx] for s in batch_idx])
                inp = torch.from_numpy(batch_inp).to(device)
                with torch.no_grad():
                    out = model(inp)
                out_np = out.cpu().numpy()
                for j, s in enumerate(batch_idx):
                    preds[s:s + ctx] += out_np[j]
                    counts[s:s + ctx] += 1.0

            counts = np.maximum(counts, 1.0)
            preds /= counts[:, None]
            preds = np.clip(preds, 0.0, 1.0).astype(np.float32)

            if sigma > 0:
                for c in range(2):
                    preds[:, c] = gaussian_filter1d(preds[:, c], sigma=sigma)
                    preds[:, c] = np.clip(preds[:, c], 0.0, 1.0)

            all_predictions[sid] = preds
            if (i + 1) % 25 == 0 or (i + 1) == len(test_ids):
                LOGGER.info("  %d/%d (%.1fs)", i + 1, len(test_ids), time_mod.time() - t0)

        # Build submission
        sub = sample_sub.copy()
        idx_arr = sub["index_pos"].values.astype(np.float64)
        mrp_arr = sub["mrp_pos"].values.astype(np.float64)

        for sid in test_ids:
            if sid not in all_predictions:
                continue
            pred = all_predictions[sid]
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

        sigma_str = f"sigma{sigma}" if sigma > 0 else "raw"
        fname = f"submission_gru_full312_ctx800_seed44_{sigma_str}.csv"
        sub.to_csv(config.results_dir / fname, index=False)
        LOGGER.info("Saved: %s", fname)

    output_vol.commit()
    LOGGER.info("Done! Download: modal volume get phase2-outputs-full312 results/ .")


@app.local_entrypoint()
def main():
    train_full312.remote()
