"""Pretraining script for the masked autoencoder (Pillar 2)."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

try:
    import optuna
    from optuna.exceptions import TrialPruned
except Exception:  # pragma: no cover - optional dependency path
    optuna = None
    TrialPruned = RuntimeError

import metric
from config import (
    Config,
    ensure_output_dirs,
    get_config,
    load_sweep_overrides,
    preflight_validate_submission_indices,
    set_global_seeds,
    setup_logging,
)
from data_utils import (
    build_pseudo_solution_df,
    build_submission_like_df_from_dense,
    create_artificial_mask,
    get_context_target_index,
    get_train_session_ids,
    load_metadata,
    load_train_session,
    zscore_session,
)
from pillar2_mae_model import MaskedAutoencoder

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainSessionData:
    """In-memory train session arrays and normalization parameters."""

    session_id: str
    day: int
    sbp_raw: np.ndarray
    sbp_z: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    channel_vars_raw: np.ndarray
    trial_starts: np.ndarray
    trial_ends: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.sbp_raw.shape[0])


@dataclass
class Corpus:
    """Collection of normalized training sessions."""

    sessions: dict[str, TrainSessionData]
    train_ids: list[str]
    val_ids: list[str]



def compute_visible_token_budget(context_bins: int, n_channels: int, masked_channels: int) -> int:
    """Return visible token count per sample for the current MAE masking scheme."""
    return int(context_bins) * int(n_channels - masked_channels)



def log_and_validate_token_budget(config: Config) -> int:
    """Warn/fail when the MAE visible token budget exceeds configured thresholds."""
    visible_tokens = compute_visible_token_budget(
        context_bins=config.mae_context_bins,
        n_channels=config.expected_n_channels,
        masked_channels=config.masked_channels_per_bin,
    )
    if visible_tokens > config.mae_max_visible_tokens_warn:
        LOGGER.warning(
            "Visible token budget is high (%d > warn=%d). Training may slow down.",
            visible_tokens,
            config.mae_max_visible_tokens_warn,
        )
    if visible_tokens > config.mae_max_visible_tokens_hard:
        raise ValueError(
            f"Visible token budget {visible_tokens} exceeds hard limit {config.mae_max_visible_tokens_hard}"
        )
    return visible_tokens



def _gather_windows_numpy(sbp: np.ndarray, center_bins: np.ndarray, context_bins: int) -> np.ndarray:
    """Gather edge-padded context windows around arbitrary center bins.

    Args:
        sbp: Array `(N, C)`.
        center_bins: Array `(B,)` of center indices.
        context_bins: Window length.

    Returns:
        Array `(B, context_bins, C)`.
    """
    sbp = np.asarray(sbp, dtype=np.float32)
    center_bins = np.asarray(center_bins, dtype=np.int64)
    target_idx = get_context_target_index(context_bins)
    left = target_idx
    right = context_bins - 1 - target_idx
    padded = np.pad(sbp, ((left, right), (0, 0)), mode="edge")
    idx = center_bins[:, None] + np.arange(context_bins)[None, :]
    return padded[idx]



def _apply_gain_augmentation(windows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply per-channel multiplicative gains in [0.5, 1.5]."""
    gains = rng.uniform(0.5, 1.5, size=(windows.shape[0], 1, windows.shape[2])).astype(np.float32)
    return (windows * gains).astype(np.float32)



def _sample_center_bins_for_session(session: TrainSessionData, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(low=0, high=session.n_bins, size=batch_size, endpoint=False, dtype=np.int64)



def _sample_channel_masks(batch_size: int, n_channels: int, n_mask: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((batch_size, n_channels), dtype=bool)
    for i in range(batch_size):
        idx = rng.choice(n_channels, size=n_mask, replace=False)
        mask[i, idx] = True
    return mask



def _zero_masked_channels_in_windows(windows: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = windows.copy()
    out[:, :, masks.shape[1] * [False]]  # no-op to validate dims; removed by optimizer
    for b in range(out.shape[0]):
        out[b, :, masks[b]] = 0.0
    return out.astype(np.float32)



def _sample_pretrain_batch(
    corpus: Corpus,
    session_ids: list[str],
    config: Config,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, Any]:
    """Sample a mixed-session pretraining batch.

    The batch samples sessions uniformly (not proportional to length) to reduce long-session dominance.
    """
    batch_size = config.mae_batch_size
    n_mask = config.masked_channels_per_bin
    n_channels = config.expected_n_channels
    context_bins = config.mae_context_bins

    chosen_session_ids = [session_ids[int(rng.integers(0, len(session_ids)))] for _ in range(batch_size)]
    windows = np.empty((batch_size, context_bins, n_channels), dtype=np.float32)
    masks = _sample_channel_masks(batch_size, n_channels, n_mask, rng)
    channel_vars = np.empty((batch_size, n_channels), dtype=np.float32)
    session_days = np.empty((batch_size,), dtype=np.float32)

    for i, sid in enumerate(chosen_session_ids):
        sess = corpus.sessions[sid]
        center = _sample_center_bins_for_session(sess, 1, rng)[0]
        windows[i] = _gather_windows_numpy(sess.sbp_z, np.array([center], dtype=np.int64), context_bins)[0]
        channel_vars[i] = np.clip(sess.channel_vars_raw.astype(np.float32), 1e-6, None)
        session_days[i] = float(sess.day)

    windows = _apply_gain_augmentation(windows, rng)
    model_in = windows.copy()
    for i in range(batch_size):
        model_in[i, :, masks[i]] = 0.0

    batch = {
        "model_in": torch.from_numpy(model_in).to(device=device, dtype=torch.float32),
        "target_windows": torch.from_numpy(windows).to(device=device, dtype=torch.float32),
        "mask": torch.from_numpy(masks).to(device=device, dtype=torch.bool),
        "channel_vars": torch.from_numpy(channel_vars).to(device=device, dtype=torch.float32),
        "session_ids": chosen_session_ids,
        "session_days": torch.from_numpy(session_days).to(device=device, dtype=torch.float32),
    }
    return batch



def _model_forward_loss(model: MaskedAutoencoder, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(
        sbp_window=batch["model_in"],
        mask=batch["mask"],
        session_ids=batch["session_ids"],
    )
    loss, metrics = model.compute_loss(
        predictions=out["pred_values"],
        targets=out["target_values"],
        masked_idx=out["masked_channel_idx"],
        channel_vars=batch["channel_vars"],
        padding_mask=out["masked_padding_mask"],
    )
    return loss, metrics



def _cosine_lr_factor(epoch: int, total_epochs: int, warmup_epochs: int, min_factor: float) -> float:
    if total_epochs <= 0:
        return 1.0
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(warmup_epochs, 1))
    if total_epochs == warmup_epochs:
        return 1.0
    progress = (epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine



def _save_checkpoint(
    path: Path,
    model: MaskedAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Config,
    best_val_nmse: float,
    train_session_ids: list[str],
    train_session_days: list[int],
) -> None:
    """Save a resumable training checkpoint."""
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": config.as_dict(),
        "best_val_nmse": best_val_nmse,
        "train_session_ids": train_session_ids,
        "train_session_days": train_session_days,
        "rng_state_python": random.getstate(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_torch": torch.random.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, path)



def _load_checkpoint_for_resume(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint dictionary for resume."""
    return torch.load(path, map_location=device)



def _select_validation_sessions(config: Config, n_val: int = 20) -> tuple[list[str], list[str]]:
    """Select day-stratified validation sessions from training metadata.

    Returns:
        `(train_ids, val_ids)`
    """
    md = load_metadata(config)
    train_md = md.loc[md["split"] == "train", ["session_id", "day"]].copy()
    train_md["day"] = train_md["day"].astype(int)
    train_md = train_md.sort_values("day").reset_index(drop=True)

    if len(train_md) <= n_val:
        raise ValueError(f"Requested {n_val} validation sessions but only {len(train_md)} train sessions available")

    # Day-stratified selection using quantile bins.
    train_md["day_bin"] = pd.qcut(train_md["day"], q=min(5, len(train_md)), labels=False, duplicates="drop")
    val_rows: list[pd.DataFrame] = []
    per_bin = max(1, n_val // int(train_md["day_bin"].nunique()))
    rng = np.random.default_rng(config.seed)
    for _, group in train_md.groupby("day_bin"):
        take = min(per_bin, len(group))
        idx = rng.choice(len(group), size=take, replace=False)
        val_rows.append(group.iloc[np.sort(idx)])
    val_md = pd.concat(val_rows, axis=0).drop_duplicates(subset=["session_id"]) if val_rows else train_md.iloc[:0]
    if len(val_md) < n_val:
        remaining = train_md.loc[~train_md["session_id"].isin(val_md["session_id"])]
        need = n_val - len(val_md)
        take_idx = rng.choice(len(remaining), size=need, replace=False)
        val_md = pd.concat([val_md, remaining.iloc[np.sort(take_idx)]], axis=0)
    elif len(val_md) > n_val:
        val_md = val_md.iloc[:n_val]

    val_ids = sorted(val_md["session_id"].astype(str).tolist())
    train_ids = sorted(train_md.loc[~train_md["session_id"].isin(val_ids), "session_id"].astype(str).tolist())
    return train_ids, val_ids



def load_train_corpus(config: Config, n_val: int = 20) -> Corpus:
    """Load and normalize all train sessions into memory."""
    all_train_ids = get_train_session_ids(config)
    train_ids, val_ids = _select_validation_sessions(config, n_val=n_val)
    # Ensure no session lost if metadata selection differs from file discovery ordering.
    selected = set(train_ids) | set(val_ids)
    missing = [sid for sid in all_train_ids if sid not in selected]
    if missing:
        train_ids.extend(sorted(missing))

    sessions: dict[str, TrainSessionData] = {}
    for sid in tqdm(sorted(selected | set(missing)), desc="Load train sessions"):
        session = load_train_session(sid, config)
        sbp_raw = np.asarray(session["sbp"], dtype=np.float32)
        sbp_z, means, stds = zscore_session(sbp_raw, return_params=True)
        sessions[sid] = TrainSessionData(
            session_id=sid,
            day=int(session["day"]),
            sbp_raw=sbp_raw,
            sbp_z=sbp_z,
            means=means.astype(np.float32),
            stds=stds.astype(np.float32),
            channel_vars_raw=np.clip(sbp_raw.var(axis=0).astype(np.float32), 1e-6, None),
            trial_starts=np.asarray(session["trial_starts"], dtype=np.int64),
            trial_ends=np.asarray(session["trial_ends"], dtype=np.int64),
        )

    return Corpus(sessions=sessions, train_ids=sorted(set(train_ids)), val_ids=sorted(set(val_ids)))



def build_model_from_corpus(config: Config, corpus: Corpus, device: torch.device) -> MaskedAutoencoder:
    """Instantiate the MAE model using training session IDs/days from the corpus."""
    ordered_train_ids = sorted(corpus.train_ids)
    ordered_train_days = [corpus.sessions[sid].day for sid in ordered_train_ids]
    model = MaskedAutoencoder(
        n_channels=config.expected_n_channels,
        context_bins=config.mae_context_bins,
        d_model=config.mae_d_model,
        n_encoder_layers=config.mae_n_encoder_layers,
        n_decoder_layers=config.mae_n_decoder_layers,
        n_heads=config.mae_n_heads,
        dim_ff=config.mae_dim_ff,
        dropout=config.mae_dropout,
        train_session_ids=ordered_train_ids,
        train_session_days=ordered_train_days,
    )
    model.to(device)
    return model


@torch.no_grad()
def predict_dense_with_model_for_session(
    model: MaskedAutoencoder,
    session: TrainSessionData,
    masked_sbp_z: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict a dense z-scored SBP array for artificially masked entries in one session."""
    model.eval()
    masked_sbp_z = np.asarray(masked_sbp_z, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if masked_sbp_z.shape != session.sbp_z.shape or mask.shape != session.sbp_z.shape:
        raise ValueError("Artificial masked arrays must match session SBP shape")

    rows_to_predict = np.where(mask.any(axis=1))[0].astype(np.int64)
    pred_dense = masked_sbp_z.copy()
    if rows_to_predict.size == 0:
        return pred_dense

    session_day_batch = None
    for start in range(0, len(rows_to_predict), batch_size):
        rows = rows_to_predict[start : start + batch_size]
        windows = _gather_windows_numpy(masked_sbp_z, rows, model.context_bins)
        row_masks = mask[rows]
        model_in = windows.copy()
        for i in range(model_in.shape[0]):
            model_in[i, :, row_masks[i]] = 0.0

        x = torch.from_numpy(model_in).to(device=device, dtype=torch.float32)
        m = torch.from_numpy(row_masks).to(device=device, dtype=torch.bool)
        if session_day_batch is None or session_day_batch.shape[0] != x.shape[0]:
            session_day_batch = torch.full((x.shape[0],), float(session.day), device=device, dtype=torch.float32)
        else:
            session_day_batch.fill_(float(session.day))

        out = model(sbp_window=x, mask=m, session_days=session_day_batch)
        preds = out["pred_values"].detach().cpu().numpy()
        masked_idx = out["masked_channel_idx"].detach().cpu().numpy()
        pad = out["masked_padding_mask"].detach().cpu().numpy().astype(bool)
        for i, row in enumerate(rows.tolist()):
            valid = ~pad[i]
            pred_dense[row, masked_idx[i, valid]] = preds[i, valid]

    return pred_dense.astype(np.float32)


@torch.no_grad()
def evaluate_model_nmse(
    model: MaskedAutoencoder,
    corpus: Corpus,
    val_ids: list[str],
    config: Config,
    device: torch.device,
    max_sessions: int | None = None,
) -> float:
    """Evaluate NMSE on validation sessions using artificial masking and metric.py."""
    model.eval()
    scores: list[float] = []
    chosen_val_ids = val_ids[:max_sessions] if max_sessions is not None else val_ids
    for sid in chosen_val_ids:
        sess = corpus.sessions[sid]
        nmse_masks: list[float] = []
        for k in range(max(1, min(config.ensemble_n_eval_masks, 2))):
            masked_raw, art_mask = create_artificial_mask(
                sess.sbp_raw,
                n_channels_to_mask=config.masked_channels_per_bin,
                seed=config.seed + 10_000 + k,
                trial_starts=sess.trial_starts,
                trial_ends=sess.trial_ends,
                constant_within_trial=True,
            )
            masked_z = (masked_raw - sess.means[None, :]) / sess.stds[None, :]
            pred_z = predict_dense_with_model_for_session(
                model=model,
                session=sess,
                masked_sbp_z=masked_z,
                mask=art_mask,
                batch_size=min(config.mae_batch_size, 512),
                device=device,
            )
            pred_raw = pred_z * sess.stds[None, :] + sess.means[None, :]
            sol = build_pseudo_solution_df(sid, sess.sbp_raw, art_mask)
            sub = build_submission_like_df_from_dense(art_mask, pred_raw)
            nmse_masks.append(metric.score(sol, sub, row_id_column_name="sample_id"))
        scores.append(float(np.mean(nmse_masks)))
    return float(np.mean(scores)) if scores else float("inf")



def train_one_run(
    config: Config,
    args: argparse.Namespace,
    trial: Any | None = None,
    sweep_overrides: dict[str, Any] | None = None,
) -> float:
    """Train one MAE run and return best validation NMSE.

    Args:
        config: Base config.
        args: CLI args.
        trial: Optional Optuna trial object.
        sweep_overrides: Optional hyperparameter overrides.

    Returns:
        Best validation NMSE observed.
    """
    from dataclasses import replace

    ensure_output_dirs(config)
    if sweep_overrides:
        config = replace(config, **sweep_overrides)

    visible_tokens = compute_visible_token_budget(
        config.mae_context_bins, config.expected_n_channels, config.masked_channels_per_bin
    )
    if visible_tokens > config.mae_max_visible_tokens_warn:
        LOGGER.warning("Visible token budget %d exceeds warn threshold %d", visible_tokens, config.mae_max_visible_tokens_warn)
    if visible_tokens > config.mae_max_visible_tokens_hard:
        msg = f"Visible token budget {visible_tokens} exceeds hard limit {config.mae_max_visible_tokens_hard}"
        if trial is not None and optuna is not None:
            raise TrialPruned(msg)
        raise ValueError(msg)

    set_global_seeds(config.seed)
    device = torch.device(config.device)
    corpus = load_train_corpus(config)
    model = build_model_from_corpus(config, corpus, device)
    param_count = model.count_parameters()
    LOGGER.info("Model parameters: %d", param_count)
    if param_count > 5_000_000 and config.mae_strict_param_budget:
        raise ValueError(f"Model exceeds 5M parameter budget: {param_count}")
    elif param_count > 5_000_000:
        LOGGER.warning("Model exceeds 5M parameter budget: %d", param_count)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.mae_lr, weight_decay=config.mae_weight_decay)
    start_epoch = 0
    best_val_nmse = float("inf")
    best_state: dict[str, Any] | None = None

    resume_path = Path(args.resume) if getattr(args, "resume", None) else None
    if resume_path is not None:
        ckpt = _load_checkpoint_for_resume(resume_path, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_nmse = float(ckpt.get("best_val_nmse", float("inf")))
        LOGGER.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    rng = np.random.default_rng(config.seed)
    train_ids = corpus.train_ids
    val_ids = corpus.val_ids

    ckpt_best_path = config.checkpoints_dir / "mae_pretrained.pt"
    ckpt_latest_path = config.checkpoints_dir / "mae_pretrain_latest.pt"
    epoch_ckpt_dir = config.checkpoints_dir / "pretrain_epochs"
    epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)

    effective_epochs = int(args.epochs) if args.epochs is not None else config.mae_epochs
    steps_per_epoch = int(args.max_steps_per_epoch) if args.max_steps_per_epoch is not None else config.mae_steps_per_epoch
    fast_dev = bool(getattr(args, "fast_dev_run", False))
    if fast_dev:
        effective_epochs = min(effective_epochs, 2)
        steps_per_epoch = min(steps_per_epoch, 5)

    for epoch in range(start_epoch, effective_epochs):
        lr_factor = _cosine_lr_factor(epoch, effective_epochs, config.mae_warmup_epochs, config.mae_lr_min_factor)
        for group in optimizer.param_groups:
            group["lr"] = config.mae_lr * lr_factor

        model.train()
        running_loss = 0.0
        running_mse = 0.0
        loop = tqdm(range(steps_per_epoch), desc=f"Pretrain epoch {epoch+1}/{effective_epochs}", leave=False)
        for _ in loop:
            batch = _sample_pretrain_batch(corpus, train_ids, config, rng, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _model_forward_loss(model, batch)
            loss.backward()
            if config.mae_grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.mae_grad_clip)
            optimizer.step()
            running_loss += metrics["loss"]
            running_mse += metrics["mse"]
            loop.set_postfix(loss=f"{metrics['loss']:.4f}")

        train_loss = running_loss / max(steps_per_epoch, 1)
        train_mse = running_mse / max(steps_per_epoch, 1)
        val_nmse = evaluate_model_nmse(
            model,
            corpus,
            val_ids,
            config,
            device,
            max_sessions=4 if fast_dev else 20,
        )
        LOGGER.info(
            "Epoch %d/%d | lr=%.2e | train_loss=%.6f | train_mse=%.6f | val_nmse=%.6f",
            epoch + 1,
            effective_epochs,
            optimizer.param_groups[0]["lr"],
            train_loss,
            train_mse,
            val_nmse,
        )

        if trial is not None and optuna is not None:
            trial.report(val_nmse, step=epoch)
            if trial.should_prune():
                raise TrialPruned(f"Pruned at epoch {epoch} with val_nmse={val_nmse:.6f}")

        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            best_state = copy.deepcopy(model.state_dict())
            _save_checkpoint(
                ckpt_best_path,
                model,
                optimizer,
                epoch,
                config,
                best_val_nmse,
                train_session_ids=sorted(corpus.train_ids),
                train_session_days=[corpus.sessions[sid].day for sid in sorted(corpus.train_ids)],
            )
            LOGGER.info("Saved best checkpoint to %s", ckpt_best_path)

        # Always persist a resumable "latest" checkpoint to reduce lost work on walltime timeout.
        _save_checkpoint(
            ckpt_latest_path,
            model,
            optimizer,
            epoch,
            config,
            best_val_nmse,
            train_session_ids=sorted(corpus.train_ids),
            train_session_days=[corpus.sessions[sid].day for sid in sorted(corpus.train_ids)],
        )

        if (epoch + 1) % 10 == 0 or epoch == effective_epochs - 1:
            epoch_path = epoch_ckpt_dir / f"mae_epoch_{epoch+1:03d}.pt"
            _save_checkpoint(
                epoch_path,
                model,
                optimizer,
                epoch,
                config,
                best_val_nmse,
                train_session_ids=sorted(corpus.train_ids),
                train_session_days=[corpus.sessions[sid].day for sid in sorted(corpus.train_ids)],
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return float(best_val_nmse)



def _suggest_sweep_overrides(config: Config, trial: Any) -> dict[str, Any]:
    """Suggest hyperparameters for an Optuna trial."""
    return {
        "mae_d_model": trial.suggest_categorical("mae_d_model", [64, 128, 256]),
        "mae_n_encoder_layers": trial.suggest_categorical("mae_n_encoder_layers", [4, 6, 8]),
        "mae_n_heads": trial.suggest_categorical("mae_n_heads", [4, 8]),
        "mae_dim_ff": trial.suggest_categorical("mae_dim_ff", [256, 512]),
        "mae_lr": trial.suggest_float("mae_lr", 5e-5, 5e-4, log=True),
        "mae_dropout": trial.suggest_float("mae_dropout", 0.05, 0.2),
        "mae_context_bins": trial.suggest_categorical("mae_context_bins", [4, 8, 16]),
    }



def run_optuna_sweep(args: argparse.Namespace) -> None:
    """Run Optuna hyperparameter sweep and save best params."""
    if optuna is None:
        raise RuntimeError("Optuna is not installed; cannot run --sweep")

    base_config = get_config(args.config)
    ensure_output_dirs(base_config)
    setup_logging(base_config.log_level, base_config.logs_dir / "pillar2_mae_sweep.log")
    preflight_validate_submission_indices(base_config)
    set_global_seeds(base_config.seed)

    n_trials = int(args.n_trials or 24)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name="ibci_mae_sweep")

    def objective(trial: Any) -> float:
        overrides = _suggest_sweep_overrides(base_config, trial)
        tmp_args = copy.copy(args)
        tmp_args.fast_dev_run = bool(getattr(args, "fast_dev_run", False))
        tmp_args.max_steps_per_epoch = args.max_steps_per_epoch or 20
        tmp_args.epochs = args.epochs or 6

        visible_tokens = compute_visible_token_budget(
            context_bins=int(overrides["mae_context_bins"]),
            n_channels=base_config.expected_n_channels,
            masked_channels=base_config.masked_channels_per_bin,
        )
        if visible_tokens > base_config.mae_max_visible_tokens_hard:
            raise TrialPruned(
                f"visible_tokens={visible_tokens} exceeds hard limit {base_config.mae_max_visible_tokens_hard}"
            )
        if visible_tokens > base_config.mae_max_visible_tokens_warn:
            LOGGER.warning("Trial %d high visible token budget: %d", trial.number, visible_tokens)

        val_nmse = train_one_run(base_config, tmp_args, trial=trial, sweep_overrides=overrides)
        return float(val_nmse)

    study.optimize(objective, n_trials=n_trials)

    ensure_output_dirs(base_config)
    best_params = study.best_trial.params if study.best_trial is not None else {}
    out_json = base_config.sweep_best_params_path
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_csv = base_config.results_dir / "sweep_study_summary.csv"
    trials_df.to_csv(trials_csv, index=False)
    LOGGER.info("Saved sweep results to %s and %s", out_json, trials_csv)



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain masked autoencoder for SBP reconstruction")
    parser.add_argument("--config", choices=["local", "hpc"], default="local")
    parser.add_argument("--sweep", action="store_true", help="Run Optuna hyperparameter sweep")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=None, help="Optuna trials")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mae_lr", type=float, default=None)
    parser.add_argument("--mae_warmup_epochs", type=int, default=None)
    parser.add_argument("--mae_mask_ratio", type=float, default=None)
    parser.add_argument("--mae_weight_decay", type=float, default=None)
    parser.add_argument("--mae_dropout", type=float, default=None)
    parser.add_argument("--mae_context_bins", type=int, default=None)
    parser.add_argument("--mae_batch_size", type=int, default=None)
    return parser.parse_args()


def _apply_pretrain_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply explicit pretraining hyperparameter overrides from CLI args."""
    updates: dict[str, Any] = {}
    for key in (
        "mae_lr",
        "mae_warmup_epochs",
        "mae_mask_ratio",
        "mae_weight_decay",
        "mae_dropout",
        "mae_context_bins",
        "mae_batch_size",
    ):
        value = getattr(args, key, None)
        if value is not None:
            updates[key] = value
    if not updates:
        return config
    LOGGER.info("Applying CLI pretrain overrides: %s", ", ".join(f"{k}={v}" for k, v in sorted(updates.items())))
    return replace(config, **updates)


def train_main(args: argparse.Namespace) -> None:
    """Main entry point for a normal pretraining run."""
    config = get_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=int(args.seed))
    config = load_sweep_overrides(config)
    config = _apply_pretrain_cli_overrides(config, args)
    ensure_output_dirs(config)
    setup_logging(config.log_level, config.logs_dir / "pillar2_mae_train.log")
    preflight_validate_submission_indices(config)
    log_and_validate_token_budget(config)
    best_val_nmse = train_one_run(config, args)
    LOGGER.info("Training complete. Best validation NMSE: %.6f", best_val_nmse)



def main() -> None:
    """CLI dispatch for pretraining and sweep modes."""
    args = _parse_args()
    if args.sweep:
        run_optuna_sweep(args)
    else:
        train_main(args)


if __name__ == "__main__":
    main()
