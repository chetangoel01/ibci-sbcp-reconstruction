"""Data utilities for Phase 2 kinematic decoding."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase2_config import Phase2Config


@dataclass
class Phase2TrainSession:
    """Train-session arrays and normalization state."""

    session_id: str
    day: int
    sbp_raw: np.ndarray
    sbp_z: np.ndarray
    kinematics: np.ndarray
    active_channels: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    trial_starts: np.ndarray
    trial_ends: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.sbp_raw.shape[0])


@dataclass
class Phase2TestSession:
    """Test-session arrays and normalization state."""

    session_id: str
    day: int
    sbp_raw: np.ndarray
    sbp_z: np.ndarray
    active_channels: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    trial_starts: np.ndarray
    trial_ends: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.sbp_raw.shape[0])


def _load_csv_cached(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


_load_csv_cached = functools.lru_cache(maxsize=8)(_load_csv_cached)


def load_metadata(config: Phase2Config) -> pd.DataFrame:
    df = _load_csv_cached(str(config.metadata_path)).copy()
    required = {"session_id", "day", "split", "n_bins", "n_trials"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Phase 2 metadata.csv missing columns: {sorted(missing)}")
    return df


def load_submission_template(config: Phase2Config) -> pd.DataFrame:
    df = _load_csv_cached(str(config.sample_sub_path)).copy()
    if "sample_id" not in df.columns:
        raise ValueError("sample_submission.csv must include sample_id")

    if {"session_id", "time_bin"} - set(df.columns):
        if config.test_index_path is None:
            raise ValueError(
                "sample_submission.csv is missing session_id/time_bin and no test_index.csv is available"
            )
        test_index = _load_csv_cached(str(config.test_index_path)).copy()
        required_index = {"sample_id", "session_id", "time_bin"}
        missing_index = required_index - set(test_index.columns)
        if missing_index:
            raise ValueError(f"test_index.csv missing columns: {sorted(missing_index)}")
        df = test_index.merge(df, on="sample_id", how="left")

    for col in ["index_pos", "mrp_pos"]:
        if col not in df.columns:
            df[col] = 0.0

    required = {"sample_id", "session_id", "time_bin", "index_pos", "mrp_pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Phase 2 sample_submission.csv missing columns: {sorted(missing)}")
    return df


def get_train_session_ids(config: Phase2Config) -> list[str]:
    md = load_metadata(config)
    out = sorted(md.loc[md["split"] == "train", "session_id"].astype(str).tolist())
    if not out:
        raise ValueError("No Phase 2 train sessions found")
    return out


def get_test_session_ids(config: Phase2Config) -> list[str]:
    md = load_metadata(config)
    out = sorted(md.loc[md["split"] == "test", "session_id"].astype(str).tolist())
    if not out:
        raise ValueError("No Phase 2 test sessions found")
    return out


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def normalize_trial_ends(starts: np.ndarray, ends_raw: np.ndarray, n_bins: int | None) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    ends_raw = np.asarray(ends_raw, dtype=np.int64)
    if starts.size == 0:
        return ends_raw.copy()
    if n_bins is not None and np.any(ends_raw > n_bins):
        raise ValueError("Trial ends exceed session length")
    if n_bins is not None and np.any(ends_raw == n_bins):
        ends = ends_raw.copy()
    elif n_bins is not None and np.all(ends_raw < n_bins):
        ends = ends_raw + 1
    else:
        ends = ends_raw + 1
    if np.any(ends <= starts):
        raise ValueError("Non-positive trial width after normalization")
    return ends.astype(np.int64)


def _load_trial_info(path: Path, n_bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(_require_file(path)) as data:
        keys = set(data.files)
        start_key = "start_bins" if "start_bins" in keys else "trial_starts" if "trial_starts" in keys else None
        end_key = "end_bins" if "end_bins" in keys else "trial_ends" if "trial_ends" in keys else None
        if start_key is None or end_key is None:
            raise ValueError(f"Trial info file {path} missing start/end bins. Found keys: {sorted(keys)}")
        starts = np.asarray(data[start_key], dtype=np.int64)
        ends_raw = np.asarray(data[end_key], dtype=np.int64)
    ends = normalize_trial_ends(starts, ends_raw, n_bins=n_bins)
    return starts, ends


def detect_active_channels(sbp: np.ndarray) -> np.ndarray:
    sbp = np.asarray(sbp, dtype=np.float32)
    if sbp.ndim != 2:
        raise ValueError(f"Expected SBP array (N, C), got {sbp.shape}")
    return ~(sbp == 0.0).all(axis=0)


def zscore_active_channels(
    sbp: np.ndarray,
    active_channels: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize only active channels and keep dead channels pinned at zero."""
    sbp = np.asarray(sbp, dtype=np.float32)
    if sbp.ndim != 2:
        raise ValueError(f"Expected SBP array (N, C), got {sbp.shape}")

    active = detect_active_channels(sbp) if active_channels is None else np.asarray(active_channels, dtype=bool)
    if active.shape != (sbp.shape[1],):
        raise ValueError(f"active_channels must have shape ({sbp.shape[1]},), got {active.shape}")

    means = np.zeros((sbp.shape[1],), dtype=np.float32)
    stds = np.ones((sbp.shape[1],), dtype=np.float32)
    out = np.zeros_like(sbp, dtype=np.float32)
    if np.any(active):
        means[active] = sbp[:, active].mean(axis=0).astype(np.float32)
        stds_active = sbp[:, active].std(axis=0).astype(np.float32)
        stds_active = np.clip(stds_active, eps, None)
        stds[active] = stds_active
        out[:, active] = (sbp[:, active] - means[active]) / stds[active]
    return out.astype(np.float32), means, stds, active


def _validate_shapes(
    sbp: np.ndarray,
    trial_starts: np.ndarray,
    trial_ends: np.ndarray,
    config: Phase2Config,
    kinematics: np.ndarray | None = None,
) -> None:
    if sbp.ndim != 2 or sbp.shape[1] != config.expected_n_channels:
        raise ValueError(f"Expected SBP shape (N, {config.expected_n_channels}), got {sbp.shape}")
    if kinematics is not None:
        if kinematics.ndim != 2 or kinematics.shape != (sbp.shape[0], config.expected_kinematic_dims):
            raise ValueError(
                f"Expected kinematics shape ({sbp.shape[0]}, {config.expected_kinematic_dims}), got {kinematics.shape}"
            )
    if trial_starts.ndim != 1 or trial_ends.ndim != 1 or trial_starts.shape != trial_ends.shape:
        raise ValueError("Invalid trial arrays")


def load_train_session(session_id: str, config: Phase2Config) -> Phase2TrainSession:
    sbp = np.load(_require_file(config.train_dir / f"{session_id}_sbp.npy")).astype(np.float32)
    kin = np.load(_require_file(config.train_dir / f"{session_id}_kinematics.npy")).astype(np.float32)
    starts, ends = _load_trial_info(config.train_dir / f"{session_id}_trial_info.npz", n_bins=sbp.shape[0])
    _validate_shapes(sbp, starts, ends, config, kinematics=kin)
    sbp_z, means, stds, active = zscore_active_channels(sbp)
    day = int(load_metadata(config).set_index("session_id").at[session_id, "day"])
    return Phase2TrainSession(
        session_id=session_id,
        day=day,
        sbp_raw=sbp,
        sbp_z=sbp_z,
        kinematics=kin,
        active_channels=active,
        means=means,
        stds=stds,
        trial_starts=starts,
        trial_ends=ends,
    )


def load_test_session(session_id: str, config: Phase2Config) -> Phase2TestSession:
    sbp = np.load(_require_file(config.test_dir / f"{session_id}_sbp.npy")).astype(np.float32)
    starts, ends = _load_trial_info(config.test_dir / f"{session_id}_trial_info.npz", n_bins=sbp.shape[0])
    _validate_shapes(sbp, starts, ends, config)
    sbp_z, means, stds, active = zscore_active_channels(sbp)
    day = int(load_metadata(config).set_index("session_id").at[session_id, "day"])
    return Phase2TestSession(
        session_id=session_id,
        day=day,
        sbp_raw=sbp,
        sbp_z=sbp_z,
        active_channels=active,
        means=means,
        stds=stds,
        trial_starts=starts,
        trial_ends=ends,
    )


def get_context_target_index(context_bins: int) -> int:
    if context_bins <= 0:
        raise ValueError("context_bins must be positive")
    return context_bins // 2 if context_bins % 2 == 1 else (context_bins // 2 - 1)


def gather_windows_numpy(values: np.ndarray, center_bins: np.ndarray, context_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    center_bins = np.asarray(center_bins, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError(f"Expected values shape (N, C), got {values.shape}")
    target_idx = get_context_target_index(context_bins)
    left = target_idx
    right = context_bins - 1 - target_idx
    padded = np.pad(values, ((left, right), (0, 0)), mode="edge")
    gather_idx = center_bins[:, None] + np.arange(context_bins)[None, :]
    return padded[gather_idx]


def clip_position_array(predictions: np.ndarray) -> np.ndarray:
    out = np.asarray(predictions, dtype=np.float32).copy()
    out[:, :2] = np.clip(out[:, :2], 0.0, 1.0)
    return out


def build_submission_from_dense_predictions(
    predictions_by_session: dict[str, np.ndarray],
    submission_template: pd.DataFrame,
) -> pd.DataFrame:
    """Merge dense per-session predictions into the Kaggle submission schema."""
    parts: list[pd.DataFrame] = []
    for session_id, dense_pred in predictions_by_session.items():
        dense_pred = np.asarray(dense_pred, dtype=np.float32)
        if dense_pred.ndim != 2 or dense_pred.shape[1] < 2:
            raise ValueError(f"Dense predictions for {session_id} must have shape (N, >=2), got {dense_pred.shape}")
        parts.append(
            pd.DataFrame(
                {
                    "session_id": session_id,
                    "time_bin": np.arange(dense_pred.shape[0], dtype=np.int64),
                    "index_pos": dense_pred[:, 0].astype(np.float32),
                    "mrp_pos": dense_pred[:, 1].astype(np.float32),
                }
            )
        )

    pred_df = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(columns=["session_id", "time_bin", "index_pos", "mrp_pos"])
    base = submission_template[["sample_id", "session_id", "time_bin"]].copy()
    merged = base.merge(pred_df, on=["session_id", "time_bin"], how="left")
    if merged[["index_pos", "mrp_pos"]].isna().any().any():
        missing = int(merged[["index_pos", "mrp_pos"]].isna().any(axis=1).sum())
        raise ValueError(f"Submission is missing predictions for {missing} rows")
    if not np.isfinite(merged[["index_pos", "mrp_pos"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("Submission contains non-finite predictions")
    return merged[["sample_id", "session_id", "time_bin", "index_pos", "mrp_pos"]].sort_values("sample_id").reset_index(drop=True)


__all__ = [
    "Phase2TestSession",
    "Phase2TrainSession",
    "build_submission_from_dense_predictions",
    "clip_position_array",
    "detect_active_channels",
    "gather_windows_numpy",
    "get_context_target_index",
    "get_test_session_ids",
    "get_train_session_ids",
    "load_metadata",
    "load_submission_template",
    "load_test_session",
    "load_train_session",
    "zscore_active_channels",
]
