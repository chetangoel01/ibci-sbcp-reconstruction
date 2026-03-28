"""Train GRU on MAE encoder features + raw SBP (ctx=800).

Pipeline:
1. Load Phase 1 MAE encoder (frozen)
2. Extract 128-dim features for all train/test sessions
3. Train bidirectional GRU on concatenated [z-scored SBP (96) + MAE features (128)] = 224-dim input
4. Run inference and generate submission

Usage:
    modal run --detach modal_train_mae_gru.py
"""

import modal

app = modal.App("phase2-mae-gru")

data_vol = modal.Volume.from_name("phase2-data", create_if_missing=True)
output_vol = modal.Volume.from_name("phase2-outputs-mae-gru", create_if_missing=True)

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
    .add_local_file("phase2_mae_features.py", "/root/repo/phase2_mae_features.py")
    .add_local_file("mae_pretrained.pt", "/root/repo/mae_pretrained.pt")
)


@app.function(
    image=code_image,
    gpu="A10G",
    timeout=14400,
    volumes={"/root/data": data_vol, "/root/outputs": output_vol},
)
def train_mae_gru():
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
        SessionCache, discover_session_ids, split_train_val,
        load_sbp, load_sample_submission, zscore_session, get_active_mask,
    )
    from phase2_model import GRUDecoder
    from phase2_train import compute_r2
    from phase2_mae_features import MAEEncoder, extract_session_features

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    LOGGER = logging.getLogger("mae_gru")

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
    # Step 1: Load MAE encoder
    # =========================================================================
    LOGGER.info("Loading MAE encoder from checkpoint...")
    mae_encoder = MAEEncoder.from_checkpoint("/root/repo/mae_pretrained.pt", device="cuda")
    n_mae_params = sum(p.numel() for p in mae_encoder.parameters())
    LOGGER.info("MAE encoder loaded: %d params (frozen)", n_mae_params)

    # =========================================================================
    # Step 2: Load sessions and extract MAE features
    # =========================================================================
    all_train_ids = discover_session_ids(config.train_dir)
    train_ids, val_ids = split_train_val(all_train_ids, n_val=config.val_sessions, seed=config.seed)
    test_ids = discover_session_ids(config.test_dir)
    LOGGER.info("Train: %d, Val: %d, Test: %d sessions", len(train_ids), len(val_ids), len(test_ids))

    # Load caches (z-scored SBP + kinematics for train/val)
    LOGGER.info("Loading training sessions...")
    train_cache = SessionCache(config.train_dir, train_ids, has_kinematics=True)
    LOGGER.info("Loading validation sessions...")
    val_cache = SessionCache(config.train_dir, val_ids, has_kinematics=True)

    # Extract MAE features for all sessions
    # Use raw (un-z-scored) SBP since the MAE was trained on raw values
    mae_features: dict[str, np.ndarray] = {}

    LOGGER.info("Extracting MAE features for train sessions...")
    t0 = time_mod.time()
    for i, sid in enumerate(train_ids):
        raw_sbp = load_sbp(config.train_dir, sid)
        active = get_active_mask(raw_sbp)
        mae_features[sid] = extract_session_features(mae_encoder, raw_sbp, active, device)
        if (i + 1) % 20 == 0:
            LOGGER.info("  Train features: %d/%d (%.1fs)", i + 1, len(train_ids), time_mod.time() - t0)
    LOGGER.info("Train features done: %d sessions (%.1fs)", len(train_ids), time_mod.time() - t0)

    LOGGER.info("Extracting MAE features for val sessions...")
    for sid in val_ids:
        raw_sbp = load_sbp(config.train_dir, sid)
        active = get_active_mask(raw_sbp)
        mae_features[sid] = extract_session_features(mae_encoder, raw_sbp, active, device)
    LOGGER.info("Val features done")

    LOGGER.info("Extracting MAE features for test sessions...")
    test_raw_sbp: dict[str, np.ndarray] = {}
    test_zscored: dict[str, np.ndarray] = {}
    test_active: dict[str, np.ndarray] = {}
    for i, sid in enumerate(test_ids):
        raw_sbp = load_sbp(config.test_dir, sid)
        active = get_active_mask(raw_sbp)
        z_sbp, _, _ = zscore_session(raw_sbp)
        mae_features[sid] = extract_session_features(mae_encoder, raw_sbp, active, device)
        test_zscored[sid] = z_sbp
        test_active[sid] = active
        if (i + 1) % 20 == 0:
            LOGGER.info("  Test features: %d/%d", i + 1, len(test_ids))
    LOGGER.info("Test features done")

    # Free MAE encoder GPU memory
    del mae_encoder
    torch.cuda.empty_cache()

    # =========================================================================
    # Step 3: Custom Dataset with MAE features
    # =========================================================================
    class MAEGRUDataset(Dataset):
        """Sliding window dataset with MAE features concatenated to z-scored SBP.

        Augmentation (gain, dropout, noise) only applies to raw SBP channels.
        MAE features are left unchanged.
        """

        def __init__(self, cache, session_ids, mae_feats, context_bins, stride,
                     samples_per_epoch=None, augment=False):
            self.cache = cache
            self.mae_feats = mae_feats
            self.context_bins = context_bins
            self.samples_per_epoch = samples_per_epoch
            self.augment = augment

            self.windows = []
            for sid in session_ids:
                n_bins = cache.sbp[sid].shape[0]
                for start in range(0, n_bins - context_bins + 1, stride):
                    self.windows.append((sid, start))

        def __len__(self):
            return self.samples_per_epoch if self.samples_per_epoch else len(self.windows)

        def __getitem__(self, idx):
            if self.samples_per_epoch:
                idx = np.random.randint(len(self.windows))

            sid, start = self.windows[idx]
            end = start + self.context_bins

            sbp = self.cache.sbp[sid][start:end].copy()  # (ctx, 96) z-scored
            active_mask = self.cache.active_masks[sid].copy()
            kin = self.cache.kinematics[sid][start:end]
            mae_feat = self.mae_feats[sid][start:end].copy()  # (ctx, 128)

            if self.augment:
                rng = np.random
                # Per-channel gain (SBP only)
                gains = rng.uniform(0.5, 1.5, size=(1, sbp.shape[1])).astype(np.float32)
                sbp = sbp * gains
                # Channel dropout (SBP only)
                drop_rate = rng.uniform(0.0, 0.3)
                n_active = int(active_mask.sum())
                n_drop = int(drop_rate * n_active)
                if n_drop > 0:
                    active_idx = np.where(active_mask)[0]
                    drop_idx = rng.choice(active_idx, size=n_drop, replace=False)
                    sbp[:, drop_idx] = 0.0
                    active_mask[drop_idx] = False
                # Gaussian noise (SBP only)
                noise = rng.normal(0, 0.05, size=sbp.shape).astype(np.float32)
                noise[:, ~active_mask] = 0.0
                sbp = sbp + noise

            # Concatenate: [z-scored SBP (96) | MAE features (128)] = 224
            combined = np.concatenate([sbp, mae_feat], axis=1)

            return {
                "sbp": torch.from_numpy(combined),
                "active_mask": torch.from_numpy(active_mask),
                "positions": torch.from_numpy(kin[:, :2].copy()),
                "velocities": torch.from_numpy(kin[:, 2:4].copy()),
            }

    train_ds = MAEGRUDataset(
        train_cache, train_ids, mae_features,
        context_bins=config.context_bins,
        stride=max(1, config.context_bins // 2),
        samples_per_epoch=len(train_ids) * 200,
        augment=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )

    # =========================================================================
    # Step 4: Build model (GRU with 224-dim input)
    # =========================================================================
    n_input_channels = 96 + 128  # raw SBP + MAE features
    model = GRUDecoder(
        n_channels=n_input_channels,
        d_model=config.gru_d_model,
        n_layers=config.gru_n_layers,
        dropout=config.gru_dropout,
        n_outputs=config.n_position_outputs,
    ).to(device)
    LOGGER.info("MAE-GRU model: %d params (input=%d)", model.count_parameters(), n_input_channels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
        return config.lr_min_factor + 0.5 * (1 - config.lr_min_factor) * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # =========================================================================
    # Step 5: Validation function (uses MAE features)
    # =========================================================================
    def validate_mae_gru():
        model.train(False)
        ctx = config.context_bins
        all_r2 = []

        with torch.no_grad():
            for sid in val_ids:
                sbp_np = val_cache.sbp[sid]  # (n_bins, 96)
                mae_np = mae_features[sid]  # (n_bins, 128)
                kin_np = val_cache.kinematics[sid]
                n_bins = sbp_np.shape[0]

                # Concatenate for model input
                combined = np.concatenate([sbp_np, mae_np], axis=1)  # (n_bins, 224)

                preds = np.zeros((n_bins, 2), dtype=np.float32)
                counts = np.zeros(n_bins, dtype=np.float32)

                stride = max(1, ctx // 2)
                starts = list(range(0, n_bins - ctx + 1, stride))
                if starts and (starts[-1] + ctx) < n_bins:
                    starts.append(n_bins - ctx)

                for start in starts:
                    inp = torch.from_numpy(combined[start:start + ctx]).unsqueeze(0).to(device)
                    out = model(inp)
                    out_np = out.squeeze(0).cpu().numpy()
                    preds[start:start + ctx] += out_np
                    counts[start:start + ctx] += 1.0

                counts = np.maximum(counts, 1.0)
                preds /= counts[:, None]

                true_pos = kin_np[:, :2]
                for c in range(2):
                    all_r2.append(compute_r2(preds[:, c], true_pos[:, c]))

        return float(np.mean(all_r2)) if all_r2 else 0.0

    # =========================================================================
    # Step 6: Training loop
    # =========================================================================
    best_val_r2 = -float("inf")
    history = []

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time_mod.time()

        for batch in train_loader:
            sbp = batch["sbp"].to(device)  # (B, ctx, 224)
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

        val_r2 = validate_mae_gru()
        is_best = val_r2 > best_val_r2
        if is_best:
            best_val_r2 = val_r2
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_r2": val_r2, "config": config.as_dict(),
                 "model_type": "mae_gru", "n_input_channels": n_input_channels},
                config.checkpoints_dir / "best_mae_gru.pt",
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
                 "val_r2": val_r2, "config": config.as_dict(),
                 "model_type": "mae_gru", "n_input_channels": n_input_channels},
                config.checkpoints_dir / "latest_mae_gru.pt",
            )
            output_vol.commit()

    with open(config.results_dir / "train_history_mae_gru.json", "w") as f:
        json.dump(history, f, indent=2)
    LOGGER.info("Training complete. Best val R2: %.4f", best_val_r2)

    # =========================================================================
    # Step 7: Inference on test set
    # =========================================================================
    LOGGER.info("Loading best checkpoint for inference...")
    ckpt = torch.load(config.checkpoints_dir / "best_mae_gru.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    LOGGER.info("Best checkpoint: epoch %d, val_r2=%.4f", ckpt["epoch"], ckpt["val_r2"])

    sample_sub = load_sample_submission(config)
    ctx = config.context_bins

    for sigma in [0, 3]:
        LOGGER.info("Inference with sigma=%d...", sigma)
        t0 = time_mod.time()

        all_predictions = {}
        for i, sid in enumerate(test_ids):
            z_sbp = test_zscored[sid]
            mae_feat = mae_features[sid]
            combined = np.concatenate([z_sbp, mae_feat], axis=1)
            n_bins = combined.shape[0]

            preds = np.zeros((n_bins, 2), dtype=np.float64)
            counts = np.zeros(n_bins, dtype=np.float64)

            stride = max(1, ctx // 2)
            starts = list(range(0, n_bins - ctx + 1, stride))
            if starts and (starts[-1] + ctx) < n_bins:
                starts.append(n_bins - ctx)

            if not starts:
                pad_len = ctx - n_bins
                padded = np.pad(combined, ((0, pad_len), (0, 0)), mode="edge")
                inp = torch.from_numpy(padded).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp)
                all_predictions[sid] = out.squeeze(0).cpu().numpy()[:n_bins].astype(np.float32)
                continue

            mask_t = torch.from_numpy(test_active[sid]).to(device)
            for batch_start in range(0, len(starts), 32):
                batch_idx = starts[batch_start:batch_start + 32]
                batch_inp = np.stack([combined[s:s + ctx] for s in batch_idx])
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
                LOGGER.info("  Predicted %d/%d (%.1fs)", i + 1, len(test_ids), time_mod.time() - t0)

        # Build submission
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

        sigma_str = f"sigma{sigma}" if sigma > 0 else "raw"
        fname = f"submission_mae_gru_ctx800_seed44_{sigma_str}.csv"
        out_path = config.results_dir / fname
        sub.to_csv(out_path, index=False)
        LOGGER.info("Saved: %s (%d rows)", fname, len(sub))

    output_vol.commit()
    LOGGER.info("Done! Download with:")
    LOGGER.info("  modal volume get phase2-outputs-mae-gru results/ .")


@app.local_entrypoint()
def main():
    train_mae_gru.remote()
