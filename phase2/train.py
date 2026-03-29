"""Training script for Phase 2 kinematic decoding (GRU).

Usage:
    python train.py --epochs 80 --context_bins 800 --seed 44
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Phase2Config, ensure_output_dirs, get_config, set_global_seeds
from data import (
    SessionCache,
    SlidingWindowDataset,
    discover_session_ids,
    split_train_val,
)
from model import build_model

LOGGER = logging.getLogger("phase2_train")

LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute R^2 for a single (session, channel) or batched predictions."""
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def validate(
    model: nn.Module,
    val_cache: SessionCache,
    val_ids: list[str],
    config: Phase2Config,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Run full-session validation and return mean R^2."""
    model.eval()
    ctx = config.context_bins
    all_r2: list[float] = []
    per_session: dict[str, float] = {}

    with torch.no_grad():
        for sid in val_ids:
            sbp_np = val_cache.sbp[sid]
            kin_np = val_cache.kinematics[sid]
            active_np = val_cache.active_masks[sid]
            n_bins = sbp_np.shape[0]

            preds = np.zeros((n_bins, 2), dtype=np.float32)
            counts = np.zeros(n_bins, dtype=np.float32)

            stride = max(1, ctx // 2)
            starts = list(range(0, n_bins - ctx + 1, stride))
            if starts and (starts[-1] + ctx) < n_bins:
                starts.append(n_bins - ctx)

            for start in starts:
                end = start + ctx
                sbp_t = torch.from_numpy(sbp_np[start:end]).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(active_np).unsqueeze(0).to(device)

                out = model(sbp_t, active_mask=mask_t)
                out_np = out.squeeze(0).cpu().numpy()

                preds[start:end] += out_np
                counts[start:end] += 1.0

            counts = np.maximum(counts, 1.0)
            preds /= counts[:, None]

            true_pos = kin_np[:, :2]
            r2_vals = []
            for c in range(2):
                r2 = compute_r2(preds[:, c], true_pos[:, c])
                r2_vals.append(r2)
                all_r2.append(r2)

            per_session[sid] = float(np.mean(r2_vals))

    mean_r2 = float(np.mean(all_r2)) if all_r2 else 0.0
    return mean_r2, per_session


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: Phase2Config) -> None:
    set_global_seeds(config.seed)
    ensure_output_dirs(config)

    log_level = LOG_LEVELS.get(config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    device = torch.device(config.device)
    LOGGER.info("Device: %s | Model: %s", device, config.model_type)

    # --- Data ---
    all_train_ids = discover_session_ids(config.train_dir)
    LOGGER.info("Discovered %d training sessions", len(all_train_ids))

    train_ids, val_ids = split_train_val(all_train_ids, n_val=config.val_sessions, seed=config.seed)
    LOGGER.info("Train: %d sessions, Val: %d sessions", len(train_ids), len(val_ids))

    LOGGER.info("Loading training sessions...")
    train_cache = SessionCache(config.train_dir, train_ids, has_kinematics=True)
    LOGGER.info("Loading validation sessions...")
    val_cache = SessionCache(config.train_dir, val_ids, has_kinematics=True)

    train_ds = SlidingWindowDataset(
        train_cache, train_ids,
        context_bins=config.context_bins,
        stride=max(1, config.context_bins // 2),
        samples_per_epoch=len(train_ids) * 200,
        augment=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        drop_last=True,
    )

    # --- Model ---
    model = build_model(config).to(device)
    LOGGER.info("Model parameters: %s", f"{model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
        return config.lr_min_factor + 0.5 * (1 - config.lr_min_factor) * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training state ---
    best_val_r2 = -float("inf")
    history: list[dict] = []

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            sbp = batch["sbp"].to(device)
            active_mask = batch["active_mask"].to(device)
            positions = batch["positions"].to(device)
            velocities = batch["velocities"].to(device)

            pred = model(sbp, active_mask=active_mask)
            loss = nn.functional.mse_loss(pred, positions)

            if config.velocity_aux_weight > 0:
                vel_pred = pred[:, 1:, :] - pred[:, :-1, :]
                vel_true = velocities[:, :-1, :]
                vel_loss = nn.functional.mse_loss(vel_pred, vel_true)
                loss = loss + config.velocity_aux_weight * vel_loss

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # --- Validation ---
        val_r2, per_session_r2 = validate(model, val_cache, val_ids, config, device)

        is_best = val_r2 > best_val_r2
        if is_best:
            best_val_r2 = val_r2
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_r2": val_r2, "config": config.as_dict()},
                config.checkpoints_dir / f"best_{config.model_type}.pt",
            )

        LOGGER.info(
            "Epoch %3d/%d | loss=%.6f | val_R2=%.4f %s | lr=%.2e | %.1fs",
            epoch + 1, config.epochs, avg_loss, val_r2,
            " *" if is_best else "", optimizer.param_groups[0]["lr"], elapsed,
        )

        history.append({
            "epoch": epoch + 1, "train_loss": avg_loss, "val_r2": val_r2,
            "best_val_r2": best_val_r2, "lr": optimizer.param_groups[0]["lr"],
        })

        # Save latest checkpoint periodically
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.epochs:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_r2": val_r2, "config": config.as_dict()},
                config.checkpoints_dir / f"latest_{config.model_type}.pt",
            )

    # Save history
    with open(config.results_dir / f"train_history_{config.model_type}.json", "w") as f:
        json.dump(history, f, indent=2)

    LOGGER.info("Training complete. Best val R^2: %.4f", best_val_r2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Phase2Config:
    parser = argparse.ArgumentParser(description="Phase 2 GRU training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--context_bins", type=int, default=None)
    parser.add_argument("--val_sessions", type=int, default=None)
    parser.add_argument("--velocity_aux", type=float, default=None, help="Velocity auxiliary loss weight")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None, help="GRU d_model")
    parser.add_argument("--n_layers", type=int, default=None, help="GRU n_layers")
    parser.add_argument("--dropout", type=float, default=None, help="GRU dropout")
    parser.add_argument("--warmup_epochs", type=int, default=None)
    args = parser.parse_args()

    config = get_config("local")

    overrides: dict = {"model_type": "gru"}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.context_bins is not None:
        overrides["context_bins"] = args.context_bins
    if args.val_sessions is not None:
        overrides["val_sessions"] = args.val_sessions
    if args.velocity_aux is not None:
        overrides["velocity_aux_weight"] = args.velocity_aux
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.d_model is not None:
        overrides["gru_d_model"] = args.d_model
    if args.n_layers is not None:
        overrides["gru_n_layers"] = args.n_layers
    if args.dropout is not None:
        overrides["gru_dropout"] = args.dropout
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = args.warmup_epochs

    return replace(config, **overrides)


if __name__ == "__main__":
    config = parse_args()
    train(config)
