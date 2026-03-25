"""Reduced-rank regression baseline for Phase 2 kinematic decoding."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    def tqdm(iterable, **_: Any):
        return iterable

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


LOGGER = setup_logging("INFO")


@dataclass
class RRRModel:
    """Simple reduced-rank regression model with ridge regularization."""

    rank: int
    ridge_alpha: float
    x_mean: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.x_mean[None, :]) @ self.coef + self.y_mean[None, :]).astype(np.float32)


def _select_validation_sessions(config: Phase2Config, requested_val_sessions: int | None = None) -> tuple[list[str], list[str]]:
    md = load_metadata(config)
    train_md = md.loc[md["split"] == "train", ["session_id", "day"]].copy()
    train_md["day"] = train_md["day"].astype(int)
    train_md = train_md.sort_values("day").reset_index(drop=True)

    target_val = requested_val_sessions if requested_val_sessions is not None else min(config.val_sessions, max(1, len(train_md) // 5))
    if len(train_md) <= target_val:
        raise ValueError(f"Need more than {target_val} train sessions to build a validation split")

    train_md["day_bin"] = pd.qcut(train_md["day"], q=min(5, len(train_md)), labels=False, duplicates="drop")
    rng = np.random.default_rng(config.seed)
    per_bin = max(1, target_val // int(train_md["day_bin"].nunique()))

    val_rows: list[pd.DataFrame] = []
    for _, group in train_md.groupby("day_bin", sort=False):
        take = min(per_bin, len(group))
        idx = rng.choice(len(group), size=take, replace=False)
        val_rows.append(group.iloc[np.sort(idx)])

    val_md = pd.concat(val_rows, axis=0).drop_duplicates(subset=["session_id"]) if val_rows else train_md.iloc[:0]
    if len(val_md) < target_val:
        remaining = train_md.loc[~train_md["session_id"].isin(val_md["session_id"])]
        need = target_val - len(val_md)
        idx = rng.choice(len(remaining), size=need, replace=False)
        val_md = pd.concat([val_md, remaining.iloc[np.sort(idx)]], axis=0)
    elif len(val_md) > target_val:
        val_md = val_md.iloc[:target_val]

    val_ids = sorted(val_md["session_id"].astype(str).tolist())
    train_ids = sorted(train_md.loc[~train_md["session_id"].isin(val_ids), "session_id"].astype(str).tolist())
    return train_ids, val_ids


def _build_row_features(
    session: Phase2TrainSession | Phase2TestSession,
    rows: np.ndarray,
    context_bins: int,
    day_mean: float,
    day_std: float,
) -> np.ndarray:
    windows = gather_windows_numpy(session.sbp_z, rows, context_bins)
    flat = windows.reshape(windows.shape[0], -1).astype(np.float32)
    active = np.broadcast_to(session.active_channels[None, :].astype(np.float32), (len(rows), session.active_channels.shape[0]))
    day_feat = np.full((len(rows), 1), (float(session.day) - day_mean) / day_std, dtype=np.float32)
    return np.concatenate([flat, active, day_feat], axis=1).astype(np.float32)


def _sample_training_matrix(
    sessions: dict[str, Phase2TrainSession],
    train_ids: list[str],
    context_bins: int,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    day_values = np.array([sessions[sid].day for sid in train_ids], dtype=np.float32)
    day_mean = float(day_values.mean())
    day_std = float(day_values.std()) if float(day_values.std()) > 1e-6 else 1.0

    total_bins = sum(sessions[sid].n_bins for sid in train_ids)
    rows_per_session: dict[str, int] = {}
    remaining = max_rows
    for i, sid in enumerate(train_ids):
        sess = sessions[sid]
        if i == len(train_ids) - 1:
            take = remaining
        else:
            frac = sess.n_bins / total_bins
            take = max(1, int(round(max_rows * frac)))
            take = min(take, remaining - max(0, len(train_ids) - i - 1))
        rows_per_session[sid] = min(take, sess.n_bins)
        remaining -= rows_per_session[sid]

    rng = np.random.default_rng(123 + max_rows + context_bins)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for sid in tqdm(train_ids, desc="Sample RRR train rows"):
        sess = sessions[sid]
        take = rows_per_session[sid]
        idx = np.sort(rng.choice(sess.n_bins, size=take, replace=False))
        x_parts.append(_build_row_features(sess, idx, context_bins, day_mean, day_std))
        y_parts.append(sess.kinematics[idx, :2].astype(np.float32))

    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)
    return x, y, day_mean, day_std


def fit_rrr(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    ridge_alpha: float,
) -> RRRModel:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Invalid RRR fit arrays: {x.shape} and {y.shape}")

    x_mean = x.mean(axis=0).astype(np.float32)
    y_mean = y.mean(axis=0).astype(np.float32)
    xc = x - x_mean[None, :]
    yc = y - y_mean[None, :]

    xtx = xc.T @ xc
    if ridge_alpha > 0:
        xtx = xtx + ridge_alpha * np.eye(xtx.shape[0], dtype=np.float32)
    xty = xc.T @ yc
    b_ridge = np.linalg.solve(xtx.astype(np.float64), xty.astype(np.float64)).astype(np.float32)

    y_hat = xc @ b_ridge
    _, _, vh = np.linalg.svd(y_hat, full_matrices=False)
    eff_rank = max(1, min(rank, vh.shape[0], y.shape[1]))
    v_r = vh[:eff_rank].T.astype(np.float32)
    coef = (b_ridge @ v_r @ v_r.T).astype(np.float32)
    return RRRModel(rank=eff_rank, ridge_alpha=ridge_alpha, x_mean=x_mean, y_mean=y_mean, coef=coef)


def evaluate_rrr(
    model: RRRModel,
    sessions: dict[str, Phase2TrainSession],
    val_ids: list[str],
    context_bins: int,
    day_mean: float,
    day_std: float,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for sid in tqdm(val_ids, desc="Evaluate RRR"):
        sess = sessions[sid]
        feature_rows = _build_row_features(sess, np.arange(sess.n_bins, dtype=np.int64), context_bins, day_mean, day_std)
        pred = clip_position_array(model.predict(feature_rows))
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


def predict_test_sessions(
    model: RRRModel,
    config: Phase2Config,
    context_bins: int,
    day_mean: float,
    day_std: float,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for sid in tqdm(get_test_session_ids(config), desc="RRR predict test"):
        sess = load_test_session(sid, config)
        x = _build_row_features(sess, np.arange(sess.n_bins, dtype=np.int64), context_bins, day_mean, day_std)
        out[sid] = clip_position_array(model.predict(x))
    return out


def _artifact_paths(config: Phase2Config) -> tuple[Path, Path, Path]:
    model_path = config.checkpoints_dir / "phase2_rrr_model.npz"
    summary_path = config.results_dir / "phase2_rrr_val_session_scores.csv"
    history_path = config.results_dir / "phase2_rrr_summary.json"
    return model_path, summary_path, history_path


def save_rrr_model(model: RRRModel, path: Path, context_bins: int, day_mean: float, day_std: float) -> None:
    np.savez(
        path,
        rank=np.int64(model.rank),
        ridge_alpha=np.float32(model.ridge_alpha),
        x_mean=model.x_mean.astype(np.float32),
        y_mean=model.y_mean.astype(np.float32),
        coef=model.coef.astype(np.float32),
        context_bins=np.int64(context_bins),
        day_mean=np.float32(day_mean),
        day_std=np.float32(day_std),
    )


def load_rrr_model(path: Path) -> tuple[RRRModel, int, float, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path)
    model = RRRModel(
        rank=int(payload["rank"]),
        ridge_alpha=float(payload["ridge_alpha"]),
        x_mean=np.asarray(payload["x_mean"], dtype=np.float32),
        y_mean=np.asarray(payload["y_mean"], dtype=np.float32),
        coef=np.asarray(payload["coef"], dtype=np.float32),
    )
    return model, int(payload["context_bins"]), float(payload["day_mean"]), float(payload["day_std"])


def train_rrr_pipeline(
    config: Phase2Config,
    *,
    context_bins: int,
    rank: int,
    ridge_alpha: float,
    max_train_rows: int,
    val_sessions: int | None,
) -> tuple[Path, float]:
    ensure_phase2_output_dirs(config)
    validate_phase2_paths(config)
    set_global_seeds(config.seed)

    train_ids, val_ids = _select_validation_sessions(config, requested_val_sessions=val_sessions)
    selected = sorted(set(train_ids) | set(val_ids))
    sessions = {sid: load_train_session(sid, config) for sid in tqdm(selected, desc="Load RRR sessions")}

    x_train, y_train, day_mean, day_std = _sample_training_matrix(sessions, train_ids, context_bins, max_train_rows)
    model = fit_rrr(x_train, y_train, rank=rank, ridge_alpha=ridge_alpha)
    val_r2, summary = evaluate_rrr(model, sessions, val_ids, context_bins, day_mean, day_std)

    model_path, summary_path, history_path = _artifact_paths(config)
    save_rrr_model(model, model_path, context_bins, day_mean, day_std)
    summary.to_csv(summary_path, index=False)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rank": model.rank,
                "ridge_alpha": ridge_alpha,
                "context_bins": context_bins,
                "max_train_rows": max_train_rows,
                "val_r2": val_r2,
                "n_train_rows": int(x_train.shape[0]),
                "n_features": int(x_train.shape[1]),
            },
            f,
            indent=2,
        )
    LOGGER.info("Saved RRR model to %s", model_path)
    LOGGER.info("Phase 2 RRR validation R2: %.6f", val_r2)
    return model_path, float(val_r2)


def build_rrr_submission(config: Phase2Config, model_path: Path | None = None) -> Path:
    ensure_phase2_output_dirs(config)
    validate_phase2_paths(config)
    model_path = model_path or _artifact_paths(config)[0]
    model, context_bins, day_mean, day_std = load_rrr_model(model_path)
    predictions = predict_test_sessions(model, config, context_bins, day_mean, day_std)
    template = load_submission_template(config)
    submission = build_submission_from_dense_predictions(predictions, template)
    out_path = config.output_dir / "submission.csv"
    submission.to_csv(out_path, index=False)
    LOGGER.info("Saved Phase 2 RRR submission to %s", out_path)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 reduced-rank regression baseline")
    parser.add_argument("--config", choices=["local", "hpc"], default="local")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--context_bins", type=int, default=11)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--max_train_rows", type=int, default=250000)
    parser.add_argument("--val_sessions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--build_submission", action="store_true")
    parser.add_argument("--predict_only", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = get_phase2_config(profile=args.config, data_dir=args.data_dir, output_dir=args.output_dir)
    if args.seed is not None:
        config = replace(config, seed=int(args.seed))

    ensure_phase2_output_dirs(config)
    global LOGGER
    LOGGER = setup_logging(config.log_level, config.logs_dir / "phase2_rrr.log")

    model_path = Path(args.model_path).resolve() if args.model_path is not None else None
    if args.predict_only:
        build_rrr_submission(config, model_path=model_path)
        return

    trained_model_path, val_r2 = train_rrr_pipeline(
        config,
        context_bins=int(args.context_bins),
        rank=int(args.rank),
        ridge_alpha=float(args.ridge_alpha),
        max_train_rows=int(args.max_train_rows),
        val_sessions=int(args.val_sessions) if args.val_sessions is not None else None,
    )
    LOGGER.info("Finished RRR training. Validation R2: %.6f", val_r2)
    if args.build_submission:
        build_rrr_submission(config, model_path=model_path or trained_model_path)


if __name__ == "__main__":
    main()
