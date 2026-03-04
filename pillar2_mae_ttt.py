"""Test-time training (TTT) and inference for the masked autoencoder pillar."""

from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

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
    build_prediction_dataframe_from_dense,
    build_pseudo_solution_df,
    build_submission_like_df_from_dense,
    create_artificial_mask,
    extract_unmasked_trials,
    get_test_session_ids,
    load_metadata,
    load_test_mask_csv,
    load_test_session,
    zscore_session,
)
from pillar2_mae_model import MaskedAutoencoder
from pillar2_mae_train import _gather_windows_numpy

LOGGER = logging.getLogger(__name__)


@dataclass
class TestSessionAdaptData:
    """Normalized arrays and metadata for one test session during TTT."""

    session_id: str
    day: int
    sbp_unmasked_raw: np.ndarray
    sbp_unmasked_z: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    channel_vars_raw: np.ndarray
    trial_starts_concat: np.ndarray
    trial_ends_concat: np.ndarray


def _sanitize_non_finite_predictions(
    values: np.ndarray,
    *,
    label: str,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Replace non-finite prediction values with the z-score mean fallback."""
    arr = np.asarray(values, dtype=np.float32)
    bad = ~np.isfinite(arr)
    if bad.any():
        n_bad = int(bad.sum())
        LOGGER.warning("%s produced %d non-finite values; replacing with %.3f", label, n_bad, fill_value)
        arr = arr.copy()
        arr[bad] = np.float32(fill_value)
    return arr



def _select_train_session_embedding_metadata(config: Config, checkpoint: dict[str, Any]) -> tuple[list[str], list[int]]:
    ids = checkpoint.get("train_session_ids")
    days = checkpoint.get("train_session_days")
    if ids is not None and days is not None:
        return [str(x) for x in ids], [int(x) for x in days]

    md = load_metadata(config)
    train_md = md.loc[md["split"] == "train", ["session_id", "day"]].copy().sort_values("session_id")
    return train_md["session_id"].astype(str).tolist(), train_md["day"].astype(int).tolist()



def _apply_checkpoint_hyperparams(config: Config, checkpoint: dict[str, Any]) -> Config:
    ck_cfg = checkpoint.get("config", {}) or {}
    if not isinstance(ck_cfg, dict):
        return config
    from dataclasses import replace

    mapping = {
        "mae_d_model": "mae_d_model",
        "mae_n_encoder_layers": "mae_n_encoder_layers",
        "mae_n_decoder_layers": "mae_n_decoder_layers",
        "mae_n_heads": "mae_n_heads",
        "mae_dim_ff": "mae_dim_ff",
        "mae_dropout": "mae_dropout",
        "mae_context_bins": "mae_context_bins",
        "expected_n_channels": "expected_n_channels",
    }
    updates: dict[str, Any] = {}
    for k_src, k_dst in mapping.items():
        if k_src in ck_cfg:
            updates[k_dst] = ck_cfg[k_src]
    return replace(config, **updates) if updates else config



def load_pretrained_model(config: Config, checkpoint_path: Path, device: torch.device) -> tuple[MaskedAutoencoder, dict[str, Any], Config]:
    """Load a pretrained MAE checkpoint and instantiate the model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    runtime_config = _apply_checkpoint_hyperparams(config, checkpoint)
    train_ids, train_days = _select_train_session_embedding_metadata(runtime_config, checkpoint)

    model = MaskedAutoencoder(
        n_channels=runtime_config.expected_n_channels,
        context_bins=runtime_config.mae_context_bins,
        d_model=runtime_config.mae_d_model,
        n_encoder_layers=runtime_config.mae_n_encoder_layers,
        n_decoder_layers=runtime_config.mae_n_decoder_layers,
        n_heads=runtime_config.mae_n_heads,
        dim_ff=runtime_config.mae_dim_ff,
        dropout=runtime_config.mae_dropout,
        train_session_ids=train_ids,
        train_session_days=train_days,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint, runtime_config



def _build_concat_unmasked_trial_bounds(test_session_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(test_session_dict["trial_starts"], dtype=np.int64)
    ends = np.asarray(test_session_dict["trial_ends"], dtype=np.int64)
    mask = np.asarray(test_session_dict["mask"], dtype=bool)
    out_starts: list[int] = []
    out_ends: list[int] = []
    offset = 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        if not bool(mask[s:e].any()):
            length = e - s
            out_starts.append(offset)
            out_ends.append(offset + length)
            offset += length
    return np.asarray(out_starts, dtype=np.int64), np.asarray(out_ends, dtype=np.int64)



def prepare_test_session_adaptation_data(test_session_dict: dict[str, Any]) -> TestSessionAdaptData:
    """Prepare normalized unmasked-trial data for TTT on one test session."""
    sbp_unmasked_raw = extract_unmasked_trials(test_session_dict)
    sbp_unmasked_z, means, stds = zscore_session(sbp_unmasked_raw, return_params=True)
    concat_starts, concat_ends = _build_concat_unmasked_trial_bounds(test_session_dict)
    return TestSessionAdaptData(
        session_id=str(test_session_dict["session_id"]),
        day=int(test_session_dict["day"]),
        sbp_unmasked_raw=sbp_unmasked_raw.astype(np.float32),
        sbp_unmasked_z=sbp_unmasked_z.astype(np.float32),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
        channel_vars_raw=np.clip(sbp_unmasked_raw.var(axis=0).astype(np.float32), 1e-6, None),
        trial_starts_concat=concat_starts,
        trial_ends_concat=concat_ends,
    )



def _sample_channel_masks(batch_size: int, n_channels: int, n_mask: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((batch_size, n_channels), dtype=bool)
    for i in range(batch_size):
        idx = rng.choice(n_channels, size=n_mask, replace=False)
        mask[i, idx] = True
    return mask



def _sample_ttt_batch(
    adapt: TestSessionAdaptData,
    config: Config,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, Any]:
    batch_size = config.ttt_batch_size
    centers = rng.integers(0, adapt.sbp_unmasked_z.shape[0], size=batch_size, endpoint=False, dtype=np.int64)
    windows = _gather_windows_numpy(adapt.sbp_unmasked_z, centers, config.mae_context_bins)
    masks = _sample_channel_masks(batch_size, config.expected_n_channels, config.masked_channels_per_bin, rng)
    model_in = windows.copy()
    for i in range(batch_size):
        model_in[i, :, masks[i]] = 0.0
    return {
        "model_in": torch.from_numpy(model_in).to(device=device, dtype=torch.float32),
        "mask": torch.from_numpy(masks).to(device=device, dtype=torch.bool),
        "channel_vars": torch.from_numpy(np.repeat(adapt.channel_vars_raw[None, :], batch_size, axis=0)).to(
            device=device, dtype=torch.float32
        ),
        "session_days": torch.full((batch_size,), float(adapt.day), device=device, dtype=torch.float32),
    }



def _set_finetune_mode(model: MaskedAutoencoder, mode: str) -> None:
    """Select trainable parameters for test-time adaptation."""
    mode = mode.lower()
    for p in model.parameters():
        p.requires_grad = False

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    if mode == "decoder_only":
        for module in [model.decoder_blocks, model.decoder_norm, model.output_head, model.channel_embedding, model.mask_token]:
            if isinstance(module, torch.nn.Parameter):
                module.requires_grad = True
            else:
                for p in module.parameters():
                    p.requires_grad = True
        return

    if mode == "norm_only":
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True
        return

    raise ValueError(f"Unknown TTT finetune mode: {mode}")



def _ttt_loss_step(model: MaskedAutoencoder, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(
        sbp_window=batch["model_in"],
        mask=batch["mask"],
        session_days=batch["session_days"],
    )
    return model.compute_loss(
        predictions=out["pred_values"],
        targets=out["target_values"],
        masked_idx=out["masked_channel_idx"],
        channel_vars=batch["channel_vars"],
        padding_mask=out["masked_padding_mask"],
    )


@torch.no_grad()
def predict_dense_for_test_session(
    model: MaskedAutoencoder,
    test_session_dict: dict[str, Any],
    adapt: TestSessionAdaptData,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run MAE inference on actual masked trials and return dense raw-space predictions."""
    model.eval()
    sbp_masked_raw = np.asarray(test_session_dict["sbp_masked"], dtype=np.float32)
    actual_mask = np.asarray(test_session_dict["mask"], dtype=bool)

    sbp_masked_z = (sbp_masked_raw - adapt.means[None, :]) / adapt.stds[None, :]
    sbp_masked_z = sbp_masked_z.astype(np.float32)
    sbp_masked_z[actual_mask] = 0.0
    pred_z_dense = sbp_masked_z.copy()

    rows_to_predict = np.where(actual_mask.any(axis=1))[0].astype(np.int64)
    if rows_to_predict.size == 0:
        return sbp_masked_raw.copy()

    for start in range(0, len(rows_to_predict), batch_size):
        rows = rows_to_predict[start : start + batch_size]
        windows = _gather_windows_numpy(sbp_masked_z, rows, model.context_bins)
        row_masks = actual_mask[rows]
        # Ensure masked channels remain zero across the full window for the current row pattern.
        for i in range(windows.shape[0]):
            windows[i, :, row_masks[i]] = 0.0
        x = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
        m = torch.from_numpy(row_masks).to(device=device, dtype=torch.bool)
        days = torch.full((x.shape[0],), float(adapt.day), device=device, dtype=torch.float32)
        out = model(sbp_window=x, mask=m, session_days=days)
        preds = out["pred_values"].cpu().numpy()
        preds = _sanitize_non_finite_predictions(preds, label=f"TTT inference for {adapt.session_id}")
        idx = out["masked_channel_idx"].cpu().numpy()
        pad = out["masked_padding_mask"].cpu().numpy().astype(bool)
        for i, row in enumerate(rows.tolist()):
            valid = ~pad[i]
            pred_z_dense[row, idx[i, valid]] = preds[i, valid]

    pred_raw = pred_z_dense * adapt.stds[None, :] + adapt.means[None, :]
    return pred_raw.astype(np.float32)



def _pretrained_ckpt_path(config: Config) -> Path:
    return config.checkpoints_dir / "mae_pretrained.pt"



def _ttt_ckpt_path(config: Config, session_id: str) -> Path:
    return config.checkpoints_dir / f"mae_ttt_{session_id}.pt"



def _ttt_prediction_path(config: Config, session_id: str) -> Path:
    return config.ttt_predictions_dir / f"{session_id}.csv"



def finetune_one_session(
    session_id: str,
    config: Config,
    checkpoint_path: Path | None = None,
    overwrite: bool = False,
    fast_dev_run: bool = False,
    save_checkpoint: bool = True,
    save_predictions: bool = True,
) -> pd.DataFrame:
    """Fine-tune MAE on one test session and run inference on its masked trials.

    Args:
        session_id: Test session ID.
        config: Runtime config.
        checkpoint_path: Pretrained checkpoint path. Defaults to `checkpoints/mae_pretrained.pt`.
        overwrite: Whether to ignore cached outputs.
        fast_dev_run: If True, run very short TTT for smoke testing.
        save_checkpoint: Whether to save adapted checkpoint.
        save_predictions: Whether to save per-session prediction CSV.

    Returns:
        Prediction DataFrame for the session (rows aligned to `test_mask.csv` for that session).
    """
    ensure_output_dirs(config)
    pred_path = _ttt_prediction_path(config, session_id)
    if pred_path.exists() and not overwrite:
        LOGGER.info("Loading cached TTT predictions for %s from %s", session_id, pred_path)
        return pd.read_csv(pred_path)

    device = torch.device(config.device)
    checkpoint_path = checkpoint_path or _pretrained_ckpt_path(config)
    model, ckpt, model_config = load_pretrained_model(config, checkpoint_path, device)
    test_session = load_test_session(session_id, model_config)
    adapt = prepare_test_session_adaptation_data(test_session)

    _set_finetune_mode(model, model_config.ttt_finetune_mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(f"No trainable parameters selected for mode={model_config.ttt_finetune_mode}")

    optimizer = torch.optim.SGD(
        trainable_params,
        lr=model_config.ttt_lr,
        momentum=model_config.ttt_momentum,
        weight_decay=model_config.ttt_weight_decay,
    )
    rng = np.random.default_rng(model_config.seed + 1000 + int(session_id[1:]))

    epochs = min(model_config.ttt_epochs, 2) if fast_dev_run else model_config.ttt_epochs
    steps_per_epoch = min(model_config.ttt_steps_per_epoch, 5) if fast_dev_run else model_config.ttt_steps_per_epoch

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        running = 0.0
        loop = tqdm(range(steps_per_epoch), desc=f"TTT {session_id} {epoch+1}/{epochs}", leave=False)
        for _ in loop:
            batch = _sample_ttt_batch(adapt, model_config, rng, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _ttt_loss_step(model, batch)
            loss.backward()
            if model_config.ttt_grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable_params, max_norm=model_config.ttt_grad_clip)
            optimizer.step()
            running += metrics["loss"]
            loop.set_postfix(loss=f"{metrics['loss']:.4f}")
        avg_loss = running / max(steps_per_epoch, 1)
        LOGGER.info("TTT %s epoch %d/%d | loss=%.6f", session_id, epoch + 1, epochs, avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    if save_checkpoint:
        ttt_ckpt_path = _ttt_ckpt_path(model_config, session_id)
        torch.save(
            {
                "model_state": model.state_dict(),
                "base_checkpoint": str(checkpoint_path),
                "session_id": session_id,
                "session_day": adapt.day,
                "config": model_config.as_dict(),
                "ttt_mode": model_config.ttt_finetune_mode,
            },
            ttt_ckpt_path,
        )
        LOGGER.info("Saved TTT checkpoint for %s to %s", session_id, ttt_ckpt_path)

    pred_dense = predict_dense_for_test_session(
        model=model,
        test_session_dict=test_session,
        adapt=adapt,
        batch_size=min(model_config.ttt_batch_size, 512),
        device=device,
    )
    test_mask_df = load_test_mask_csv(model_config)
    mask_rows = test_mask_df.loc[test_mask_df["session_id"].astype(str) == session_id]
    pred_df = build_prediction_dataframe_from_dense(session_id, pred_dense, mask_rows)

    if save_predictions:
        pred_df.to_csv(pred_path, index=False)
        LOGGER.info("Saved TTT predictions for %s to %s", session_id, pred_path)
    return pred_df



def run_all_sessions(config: Config, overwrite: bool = False, fast_dev_run: bool = False) -> pd.DataFrame:
    """Run TTT for all test sessions and save consolidated MAE predictions."""
    ensure_output_dirs(config)
    session_ids = get_test_session_ids(config)
    outputs: list[pd.DataFrame] = []
    for sid in tqdm(session_ids, desc="TTT sessions"):
        outputs.append(
            finetune_one_session(
                session_id=sid,
                config=config,
                overwrite=overwrite,
                fast_dev_run=fast_dev_run,
                save_checkpoint=True,
                save_predictions=True,
            )
        )
    all_preds = pd.concat(outputs, axis=0, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    out_path = config.results_dir / "mae_predictions.csv"
    all_preds.to_csv(out_path, index=False)
    LOGGER.info("Saved consolidated MAE predictions to %s", out_path)
    return all_preds



def _load_ttt_or_pretrained_for_session(session_id: str, config: Config, device: torch.device) -> tuple[MaskedAutoencoder, Config]:
    ttt_path = _ttt_ckpt_path(config, session_id)
    base_ckpt = _pretrained_ckpt_path(config)
    model, _, model_config = load_pretrained_model(config, base_ckpt, device)
    if ttt_path.exists():
        ttt_ckpt = torch.load(ttt_path, map_location=device)
        model.load_state_dict(ttt_ckpt["model_state"])
    else:
        LOGGER.warning("TTT checkpoint missing for %s; using pretrained model for MAE weight estimation", session_id)
    model.eval()
    return model, model_config


@torch.no_grad()
def evaluate_session_nmse_on_artificial_masks(test_session_dict: dict[str, Any], config: Config) -> float:
    """Evaluate (adapted) MAE NMSE on artificial masks from a test session's unmasked trials.

    This is used by `ensemble.py` to estimate per-session ensemble weights.
    """
    session_id = str(test_session_dict["session_id"])
    device = torch.device(config.device)
    model, model_config = _load_ttt_or_pretrained_for_session(session_id, config, device)
    adapt = prepare_test_session_adaptation_data(test_session_dict)

    scores: list[float] = []
    for i in range(model_config.ensemble_n_eval_masks):
        masked_raw, art_mask = create_artificial_mask(
            adapt.sbp_unmasked_raw,
            n_channels_to_mask=model_config.masked_channels_per_bin,
            seed=model_config.seed + 20_000 + i,
            trial_starts=adapt.trial_starts_concat,
            trial_ends=adapt.trial_ends_concat,
            constant_within_trial=True,
        )
        masked_z = (masked_raw - adapt.means[None, :]) / adapt.stds[None, :]
        masked_z = masked_z.astype(np.float32)
        masked_z[art_mask] = 0.0

        pred_z = masked_z.copy()
        rows = np.where(art_mask.any(axis=1))[0].astype(np.int64)
        for start in range(0, len(rows), min(model_config.ttt_batch_size, 512)):
            r = rows[start : start + min(model_config.ttt_batch_size, 512)]
            windows = _gather_windows_numpy(masked_z, r, model.context_bins)
            row_masks = art_mask[r]
            for j in range(windows.shape[0]):
                windows[j, :, row_masks[j]] = 0.0
            x = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
            m = torch.from_numpy(row_masks).to(device=device, dtype=torch.bool)
            days = torch.full((x.shape[0],), float(adapt.day), device=device, dtype=torch.float32)
            out = model(sbp_window=x, mask=m, session_days=days)
            preds = out["pred_values"].cpu().numpy()
            preds = _sanitize_non_finite_predictions(preds, label=f"Artificial-mask eval for {session_id}")
            idx = out["masked_channel_idx"].cpu().numpy()
            pad = out["masked_padding_mask"].cpu().numpy().astype(bool)
            for j, row in enumerate(r.tolist()):
                valid = ~pad[j]
                pred_z[row, idx[j, valid]] = preds[j, valid]

        pred_raw = pred_z * adapt.stds[None, :] + adapt.means[None, :]
        sol = build_pseudo_solution_df(session_id, adapt.sbp_unmasked_raw, art_mask)
        sub = build_submission_like_df_from_dense(art_mask, pred_raw)
        scores.append(metric.score(sol, sub, row_id_column_name="sample_id"))
    return float(np.mean(scores)) if scores else float("inf")



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test-time fine-tune the MAE and run inference on test sessions")
    parser.add_argument("--config", choices=["local", "hpc"], default="local")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained checkpoint path")
    parser.add_argument("--session_id", type=str, default=None)
    parser.add_argument("--session_index", type=int, default=None)
    parser.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()



def main() -> None:
    """CLI entry point for TTT and MAE inference."""
    args = _parse_args()
    config = load_sweep_overrides(get_config(args.config))
    ensure_output_dirs(config)
    setup_logging(config.log_level, config.logs_dir / "pillar2_mae_ttt.log")
    preflight_validate_submission_indices(config)
    set_global_seeds(config.seed)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else None

    if args.run_all:
        run_all_sessions(config=config, overwrite=args.overwrite, fast_dev_run=args.fast_dev_run)
        return

    session_ids = get_test_session_ids(config)
    if args.session_id is None and args.session_index is None:
        raise ValueError("Provide --session_id, --session_index, or --all")
    if args.session_id is not None:
        session_id = args.session_id
    else:
        if args.session_index is None or not (0 <= args.session_index < len(session_ids)):
            raise ValueError(f"session_index must be in [0, {len(session_ids)-1}]")
        session_id = session_ids[args.session_index]

    finetune_one_session(
        session_id=session_id,
        config=config,
        checkpoint_path=checkpoint_path,
        overwrite=args.overwrite,
        fast_dev_run=args.fast_dev_run,
        save_checkpoint=True,
        save_predictions=True,
    )


if __name__ == "__main__":
    main()
