"""Training and submission pipeline for Phase 2 kinematic decoding."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from config import set_global_seeds, setup_logging
from phase2_config import Phase2Config, ensure_phase2_output_dirs, get_phase2_config, validate_phase2_paths
from phase2_data import (
    Phase2TestSession,
    Phase2TrainSession,
    build_submission_from_dense_predictions,
    clip_position_array,
    gather_windows_numpy,
    get_test_session_ids,
    get_train_session_ids,
    load_metadata,
    load_submission_template,
    load_test_session,
    load_train_session,
)
from phase2_metric import mean_session_r2, summarize_session_scores
from phase2_models import GRUKinematicsDecoder, TransformerKinematicsDecoder


LOGGER = setup_logging("INFO")


@dataclass
class Phase2Corpus:
    """In-memory train corpus for random window sampling."""

    sessions: dict[str, Phase2TrainSession]
    train_ids: list[str]
    val_ids: list[str]
    day_mean: float
    day_std: float


def _select_validation_sessions(config: Phase2Config) -> tuple[list[str], list[str]]:
    md = load_metadata(config)
    train_md = md.loc[md["split"] == "train", ["session_id", "day"]].copy()
    train_md["day"] = train_md["day"].astype(int)
    train_md = train_md.sort_values("day").reset_index(drop=True)

    n_val = min(config.val_sessions, max(1, len(train_md) // 5))
    if len(train_md) <= n_val:
        raise ValueError(f"Need more than {n_val} train sessions to build a validation split")

    train_md["day_bin"] = pd.qcut(train_md["day"], q=min(5, len(train_md)), labels=False, duplicates="drop")
    rng = np.random.default_rng(config.seed)
    per_bin = max(1, n_val // int(train_md["day_bin"].nunique()))

    val_rows: list[pd.DataFrame] = []
    for _, group in train_md.groupby("day_bin", sort=False):
        take = min(per_bin, len(group))
        if take <= 0:
            continue
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


def load_train_corpus(config: Phase2Config) -> Phase2Corpus:
    all_train_ids = get_train_session_ids(config)
    train_ids, val_ids = _select_validation_sessions(config)
    selected = sorted(set(train_ids) | set(val_ids))
    sessions = {sid: load_train_session(sid, config) for sid in tqdm(selected, desc="Load Phase 2 train sessions")}
    days = np.array([sessions[sid].day for sid in selected], dtype=np.float32)
    day_mean = float(days.mean())
    day_std = float(days.std()) if float(days.std()) > 1e-6 else 1.0

    missing = [sid for sid in all_train_ids if sid not in sessions]
    if missing:
        raise ValueError(f"Some train sessions were not loaded: {missing[:5]}")

    return Phase2Corpus(
        sessions=sessions,
        train_ids=train_ids,
        val_ids=val_ids,
        day_mean=day_mean,
        day_std=day_std,
    )


def _build_gru_inputs(
    windows: np.ndarray,
    active_channels: np.ndarray,
    days: np.ndarray,
    corpus: Phase2Corpus,
) -> np.ndarray:
    time_steps = windows.shape[1]
    active_feat = np.broadcast_to(
        active_channels[:, None, :].astype(np.float32),
        (windows.shape[0], time_steps, active_channels.shape[1]),
    )
    day_norm = ((days.astype(np.float32) - np.float32(corpus.day_mean)) / np.float32(corpus.day_std)).reshape(-1, 1, 1)
    day_feat = np.broadcast_to(day_norm, (windows.shape[0], time_steps, 1))
    return np.concatenate([windows.astype(np.float32), active_feat, day_feat], axis=2).astype(np.float32)


def _sample_train_batch(
    corpus: Phase2Corpus,
    config: Phase2Config,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size = config.batch_size
    context_bins = config.context_bins
    n_channels = config.expected_n_channels

    chosen_ids = [corpus.train_ids[int(rng.integers(0, len(corpus.train_ids)))] for _ in range(batch_size)]
    centers = np.empty((batch_size,), dtype=np.int64)
    windows = np.empty((batch_size, context_bins, n_channels), dtype=np.float32)
    targets = np.empty((batch_size, config.expected_kinematic_dims), dtype=np.float32)
    active = np.empty((batch_size, n_channels), dtype=bool)
    days = np.empty((batch_size,), dtype=np.float32)

    for i, sid in enumerate(chosen_ids):
        sess = corpus.sessions[sid]
        center = int(rng.integers(0, sess.n_bins))
        centers[i] = center
        windows[i] = gather_windows_numpy(sess.sbp_z, np.array([center], dtype=np.int64), context_bins)[0]
        targets[i] = sess.kinematics[center].astype(np.float32)
        active[i] = sess.active_channels
        days[i] = float(sess.day)

    return {
        "windows": torch.from_numpy(windows).to(device=device, dtype=torch.float32),
        "active_mask": torch.from_numpy(active).to(device=device, dtype=torch.bool),
        "session_days": torch.from_numpy(days).to(device=device, dtype=torch.float32),
        "targets": torch.from_numpy(targets).to(device=device, dtype=torch.float32),
    }


def _compute_loss(
    model: GRUKinematicsDecoder | TransformerKinematicsDecoder,
    batch: dict[str, torch.Tensor],
    config: Phase2Config,
    corpus: Phase2Corpus,
    aux_velocity_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if config.model_type == "gru":
        gru_inputs = _build_gru_inputs(
            batch["windows"].detach().cpu().numpy(),
            batch["active_mask"].detach().cpu().numpy(),
            batch["session_days"].detach().cpu().numpy(),
            corpus,
        )
        preds = model(torch.from_numpy(gru_inputs).to(device=batch["windows"].device, dtype=torch.float32))
    elif config.model_type == "transformer":
        preds = model(
            batch["windows"],
            batch["active_mask"],
            batch["session_days"],
        )
    else:
        raise ValueError(f"Unknown model_type={config.model_type}")
    targets = batch["targets"]
    pos_loss = torch.mean(torch.square(preds[:, :2] - targets[:, :2]))
    vel_loss = torch.mean(torch.square(preds[:, 2:] - targets[:, 2:]))
    loss = pos_loss + aux_velocity_weight * vel_loss
    return loss, {
        "loss": float(loss.detach().item()),
        "pos_loss": float(pos_loss.detach().item()),
        "vel_loss": float(vel_loss.detach().item()),
    }


def build_model(
    config: Phase2Config,
    corpus: Phase2Corpus,
    device: torch.device,
) -> GRUKinematicsDecoder | TransformerKinematicsDecoder:
    if config.model_type == "gru":
        model: GRUKinematicsDecoder | TransformerKinematicsDecoder = GRUKinematicsDecoder(
            input_dim=config.input_feature_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.expected_kinematic_dims,
            target_index=config.target_index,
        )
    elif config.model_type == "transformer":
        train_days = [corpus.sessions[sid].day for sid in corpus.train_ids]
        model = TransformerKinematicsDecoder(
            n_channels=config.expected_n_channels,
            context_bins=config.context_bins,
            d_model=config.transformer_d_model,
            n_layers=config.transformer_n_layers,
            n_heads=config.transformer_n_heads,
            dim_ff=config.transformer_dim_ff,
            dropout=config.dropout,
            output_dim=config.expected_kinematic_dims,
            train_session_ids=corpus.train_ids,
            train_session_days=train_days,
        )
    else:
        raise ValueError(f"Unknown model_type={config.model_type}")
    model.to(device)
    return model


def _apply_checkpoint_hparams(config: Phase2Config, checkpoint: dict[str, Any]) -> Phase2Config:
    saved = checkpoint.get("config", {}) or {}
    updates: dict[str, Any] = {}
    for key in [
        "model_type",
        "context_bins",
        "batch_size",
        "eval_batch_size",
        "hidden_size",
        "num_layers",
        "dropout",
        "transformer_d_model",
        "transformer_n_layers",
        "transformer_n_heads",
        "transformer_dim_ff",
        "lr",
        "weight_decay",
        "epochs",
        "steps_per_epoch",
        "val_sessions",
        "aux_velocity_weight",
        "grad_clip",
        "clip_positions",
        "expected_n_channels",
        "expected_kinematic_dims",
    ]:
        if key in saved:
            updates[key] = saved[key]
    return replace(config, **updates) if updates else config


def load_checkpoint_model(
    config: Phase2Config,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[GRUKinematicsDecoder, dict[str, Any], Phase2Config]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Phase 2 checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    runtime_config = _apply_checkpoint_hparams(config, checkpoint)
    train_ids = checkpoint.get("train_ids")
    val_ids = checkpoint.get("val_ids")
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise ValueError("Checkpoint is missing train_ids/val_ids")
    selected = sorted(set(str(x) for x in train_ids) | set(str(x) for x in val_ids))
    sessions = {sid: load_train_session(sid, runtime_config) for sid in selected}
    corpus = Phase2Corpus(
        sessions=sessions,
        train_ids=[str(x) for x in train_ids],
        val_ids=[str(x) for x in val_ids],
        day_mean=float(checkpoint.get("day_mean", 0.0)),
        day_std=float(checkpoint.get("day_std", 1.0)) or 1.0,
    )
    model = build_model(runtime_config, corpus, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, runtime_config


def _maybe_initialize_from_phase1(
    model: GRUKinematicsDecoder | TransformerKinematicsDecoder,
    checkpoint_path: Path | None,
    device: torch.device,
) -> None:
    if checkpoint_path is None or not isinstance(model, TransformerKinematicsDecoder):
        return
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", {})
    if not isinstance(state, dict):
        raise ValueError("Phase 1 checkpoint missing model_state")

    own_state = model.state_dict()
    copied: list[str] = []
    for key, value in state.items():
        if key.startswith("channel_embedding.") or key.startswith("encoder_blocks.") or key == "encoder_norm.weight" or key == "encoder_norm.bias":
            if key in own_state and own_state[key].shape == value.shape:
                own_state[key] = value
                copied.append(key)
    if copied:
        model.load_state_dict(own_state, strict=False)
        LOGGER.info("Initialized Phase 2 transformer from Phase 1 checkpoint %s (%d tensors)", checkpoint_path, len(copied))
    else:
        LOGGER.warning("No compatible transformer weights were loaded from %s", checkpoint_path)


@torch.no_grad()
def predict_session_kinematics(
    model: GRUKinematicsDecoder | TransformerKinematicsDecoder,
    session: Phase2TrainSession | Phase2TestSession,
    corpus: Phase2Corpus,
    config: Phase2Config,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out = np.empty((session.n_bins, config.expected_kinematic_dims), dtype=np.float32)
    rows = np.arange(session.n_bins, dtype=np.int64)
    for start in range(0, session.n_bins, config.eval_batch_size):
        chunk = rows[start : start + config.eval_batch_size]
        windows = gather_windows_numpy(session.sbp_z, chunk, config.context_bins)
        active = np.broadcast_to(session.active_channels[None, :], (len(chunk), config.expected_n_channels))
        days = np.full((len(chunk),), float(session.day), dtype=np.float32)
        if config.model_type == "gru":
            inputs = _build_gru_inputs(windows, active, days, corpus)
            preds = model(torch.from_numpy(inputs).to(device=device, dtype=torch.float32))
        elif config.model_type == "transformer":
            preds = model(
                torch.from_numpy(windows).to(device=device, dtype=torch.float32),
                torch.from_numpy(active).to(device=device, dtype=torch.bool),
                torch.from_numpy(days).to(device=device, dtype=torch.float32),
            )
        else:
            raise ValueError(f"Unknown model_type={config.model_type}")
        out[chunk] = preds.detach().cpu().numpy().astype(np.float32)
    return clip_position_array(out) if config.clip_positions else out


@torch.no_grad()
def evaluate_model_r2(
    model: GRUKinematicsDecoder | TransformerKinematicsDecoder,
    corpus: Phase2Corpus,
    config: Phase2Config,
    device: torch.device,
    max_sessions: int | None = None,
) -> tuple[float, pd.DataFrame]:
    chosen_val_ids = corpus.val_ids[:max_sessions] if max_sessions is not None else corpus.val_ids
    rows: list[dict[str, Any]] = []
    for sid in chosen_val_ids:
        sess = corpus.sessions[sid]
        pred = predict_session_kinematics(model, sess, corpus, config, device)
        mean_r2, per_target = mean_session_r2(sess.kinematics[:, :2], pred[:, :2])
        rows.append(
            {
                "session_id": sid,
                "day": int(sess.day),
                "r2_mean": float(mean_r2),
                "r2_index_pos": float(per_target["index_pos"]),
                "r2_mrp_pos": float(per_target["mrp_pos"]),
            }
        )
    summary = summarize_session_scores(rows)
    score = float(summary["r2_mean"].mean()) if not summary.empty else float("-inf")
    return score, summary


def _checkpoint_path(config: Phase2Config) -> Path:
    return config.checkpoints_dir / "phase2_gru_decoder.pt"


def _history_path(config: Phase2Config) -> Path:
    return config.results_dir / "phase2_train_history.json"


def _save_checkpoint(
    path: Path,
    model: GRUKinematicsDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Phase2Config,
    corpus: Phase2Corpus,
    best_val_r2: float,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_r2": best_val_r2,
        "train_ids": corpus.train_ids,
        "val_ids": corpus.val_ids,
        "day_mean": corpus.day_mean,
        "day_std": corpus.day_std,
        "config": config.as_dict(),
    }
    torch.save(payload, path)


def train_phase2_model(
    config: Phase2Config,
    fast_dev_run: bool = False,
    phase1_checkpoint: Path | None = None,
) -> tuple[Path, float]:
    ensure_phase2_output_dirs(config)
    validate_phase2_paths(config)
    set_global_seeds(config.seed)

    device = torch.device(config.device)
    corpus = load_train_corpus(config)
    model = build_model(config, corpus, device)
    _maybe_initialize_from_phase1(model, phase1_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    effective_epochs = min(config.epochs, 2) if fast_dev_run else config.epochs
    steps_per_epoch = min(config.steps_per_epoch, 5) if fast_dev_run else config.steps_per_epoch
    eval_sessions = min(4, len(corpus.val_ids)) if fast_dev_run else None

    best_val_r2 = float("-inf")
    best_state: dict[str, Any] | None = None
    best_summary = pd.DataFrame()
    history: list[dict[str, float | int]] = []
    rng = np.random.default_rng(config.seed)
    ckpt_path = _checkpoint_path(config)

    for epoch in range(effective_epochs):
        model.train()
        running_loss = 0.0
        running_pos = 0.0
        running_vel = 0.0
        loop = tqdm(range(steps_per_epoch), desc=f"Phase 2 epoch {epoch+1}/{effective_epochs}", leave=False)
        for _ in loop:
            batch = _sample_train_batch(corpus, config, rng, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _compute_loss(model, batch, config, corpus, config.aux_velocity_weight)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            running_loss += metrics["loss"]
            running_pos += metrics["pos_loss"]
            running_vel += metrics["vel_loss"]
            loop.set_postfix(loss=f"{metrics['loss']:.4f}")

        train_loss = running_loss / max(steps_per_epoch, 1)
        train_pos = running_pos / max(steps_per_epoch, 1)
        train_vel = running_vel / max(steps_per_epoch, 1)
        val_r2, summary = evaluate_model_r2(model, corpus, config, device, max_sessions=eval_sessions)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_pos_loss": train_pos,
                "train_vel_loss": train_vel,
                "val_r2": val_r2,
            }
        )
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.6f | train_pos=%.6f | train_vel=%.6f | val_r2=%.6f",
            epoch + 1,
            effective_epochs,
            train_loss,
            train_pos,
            train_vel,
            val_r2,
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = copy.deepcopy(model.state_dict())
            best_summary = summary.copy()
            _save_checkpoint(ckpt_path, model, optimizer, epoch, config, corpus, best_val_r2)
            LOGGER.info("Saved best Phase 2 checkpoint to %s", ckpt_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    with _history_path(config).open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if not best_summary.empty:
        best_summary.to_csv(config.results_dir / "phase2_val_session_scores.csv", index=False)
    return ckpt_path, float(best_val_r2)


def build_phase2_submission(
    config: Phase2Config,
    checkpoint_path: Path | None = None,
) -> Path:
    ensure_phase2_output_dirs(config)
    validate_phase2_paths(config)

    device = torch.device(config.device)
    checkpoint_path = checkpoint_path or _checkpoint_path(config)
    model, checkpoint, runtime_config = load_checkpoint_model(config, checkpoint_path, device)

    train_ids = checkpoint.get("train_ids")
    val_ids = checkpoint.get("val_ids")
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise ValueError("Checkpoint is missing train_ids/val_ids")

    selected = sorted(set(str(x) for x in train_ids) | set(str(x) for x in val_ids))
    sessions = {sid: load_train_session(sid, runtime_config) for sid in tqdm(selected, desc="Reload train corpus")}
    corpus = Phase2Corpus(
        sessions=sessions,
        train_ids=[str(x) for x in train_ids],
        val_ids=[str(x) for x in val_ids],
        day_mean=float(checkpoint.get("day_mean", 0.0)),
        day_std=float(checkpoint.get("day_std", 1.0)) or 1.0,
    )

    predictions_by_session: dict[str, np.ndarray] = {}
    for session_id in tqdm(get_test_session_ids(runtime_config), desc="Predict test sessions"):
        session = load_test_session(session_id, runtime_config)
        predictions_by_session[session_id] = predict_session_kinematics(model, session, corpus, runtime_config, device)

    submission_template = load_submission_template(runtime_config)
    submission_df = build_submission_from_dense_predictions(predictions_by_session, submission_template)
    out_path = runtime_config.output_dir / "submission.csv"
    submission_df.to_csv(out_path, index=False)
    LOGGER.info("Saved Phase 2 submission to %s", out_path)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 kinematic decoder training and submission")
    parser.add_argument("--config", choices=["local", "hpc"], default="local")
    parser.add_argument("--data_dir", type=str, default=None, help="Phase 2 dataset root")
    parser.add_argument("--output_dir", type=str, default=None, help="Output root for checkpoints/results/logs")
    parser.add_argument("--model_type", choices=["gru", "transformer"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--context_bins", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--transformer_d_model", type=int, default=None)
    parser.add_argument("--transformer_n_layers", type=int, default=None)
    parser.add_argument("--transformer_n_heads", type=int, default=None)
    parser.add_argument("--transformer_dim_ff", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--val_sessions", type=int, default=None)
    parser.add_argument("--aux_velocity_weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--phase1_checkpoint", type=str, default=None, help="Optional Phase 1 MAE checkpoint for transformer initialization")
    parser.add_argument("--build_submission", action="store_true")
    parser.add_argument("--predict_only", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def _apply_arg_overrides(config: Phase2Config, args: argparse.Namespace) -> Phase2Config:
    updates: dict[str, Any] = {}
    for key in [
        "model_type",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "context_bins",
        "hidden_size",
        "num_layers",
        "dropout",
        "transformer_d_model",
        "transformer_n_layers",
        "transformer_n_heads",
        "transformer_dim_ff",
        "lr",
        "weight_decay",
        "steps_per_epoch",
        "val_sessions",
        "aux_velocity_weight",
        "seed",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            updates[key] = value
    return replace(config, **updates) if updates else config


def main() -> None:
    args = _parse_args()
    config = get_phase2_config(profile=args.config, data_dir=args.data_dir, output_dir=args.output_dir)
    config = _apply_arg_overrides(config, args)

    ensure_phase2_output_dirs(config)
    global LOGGER
    LOGGER = setup_logging(config.log_level, config.logs_dir / "phase2_train.log")

    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint is not None else None
    if args.predict_only:
        build_phase2_submission(config, checkpoint_path=checkpoint_path)
        return

    phase1_checkpoint = Path(args.phase1_checkpoint).resolve() if args.phase1_checkpoint is not None else None
    best_checkpoint, best_val_r2 = train_phase2_model(
        config,
        fast_dev_run=args.fast_dev_run,
        phase1_checkpoint=phase1_checkpoint,
    )
    LOGGER.info("Phase 2 training complete. Best validation R2: %.6f", best_val_r2)
    if args.build_submission:
        build_phase2_submission(config, checkpoint_path=checkpoint_path or best_checkpoint)


if __name__ == "__main__":
    main()
