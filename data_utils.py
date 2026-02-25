"""Data loading and preprocessing utilities for iBCI SBP reconstruction."""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config import Config

LOGGER = logging.getLogger(__name__)


def _load_csv_cached(path_str: str) -> pd.DataFrame:
    """Cached CSV reader keyed by file path string."""
    return pd.read_csv(path_str)


_load_csv_cached = functools.lru_cache(maxsize=8)(_load_csv_cached)


def load_metadata(config: Config) -> pd.DataFrame:
    """Load `metadata.csv`.

    Args:
        config: Global configuration.

    Returns:
        DataFrame with metadata rows.
    """
    df = _load_csv_cached(str(config.metadata_path)).copy()
    required = {"session_id", "day", "split", "n_bins", "n_trials"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {sorted(missing)}")
    return df


@functools.lru_cache(maxsize=1)
def _metadata_indexed(metadata_path: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    return df.set_index("session_id", drop=False)


@functools.lru_cache(maxsize=1)
def _test_mask_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@functools.lru_cache(maxsize=1)
def _sample_sub_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)



def load_test_mask_csv(config: Config) -> pd.DataFrame:
    """Load `test_mask.csv` with cached backing read."""
    df = _test_mask_cached(str(config.test_mask_path)).copy()
    required = {"sample_id", "session_id", "time_bin", "channel"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test_mask.csv missing columns: {sorted(missing)}")
    return df



def load_sample_submission(config: Config) -> pd.DataFrame:
    """Load `sample_submission.csv` with cached backing read."""
    df = _sample_sub_cached(str(config.sample_sub_path)).copy()
    required = {"sample_id", "session_id", "time_bin", "channel", "predicted_sbp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sample_submission.csv missing columns: {sorted(missing)}")
    return df



def get_train_session_ids(config: Config) -> list[str]:
    """Return sorted training session IDs from metadata."""
    md = load_metadata(config)
    sids = sorted(md.loc[md["split"] == "train", "session_id"].astype(str).tolist())
    if not sids:
        raise ValueError("No training sessions found in metadata")
    return sids



def get_test_session_ids(config: Config) -> list[str]:
    """Return sorted test session IDs from metadata."""
    md = load_metadata(config)
    sids = sorted(md.loc[md["split"] == "test", "session_id"].astype(str).tolist())
    if len(sids) != config.expected_n_test_sessions:
        LOGGER.warning(
            "Expected %d test sessions, found %d", config.expected_n_test_sessions, len(sids)
        )
    return sids



def get_session_day(session_id: str, config: Config) -> int:
    """Return integer day index for a session ID."""
    md = _metadata_indexed(str(config.metadata_path))
    if session_id not in md.index:
        raise KeyError(f"Session {session_id} not found in metadata")
    return int(md.at[session_id, "day"])



def get_nearest_train_day(test_session_id: str, config: Config) -> tuple[str, int]:
    """Return nearest training session and absolute day gap for a test session."""
    md = load_metadata(config)
    test_day = get_session_day(test_session_id, config)
    train_rows = md.loc[md["split"] == "train", ["session_id", "day"]].copy()
    train_rows["gap"] = (train_rows["day"].astype(int) - test_day).abs()
    best = train_rows.sort_values(["gap", "day", "session_id"]).iloc[0]
    return str(best["session_id"]), int(best["gap"])



def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path



def _load_trial_info(path: Path, n_bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load and normalize trial start/end arrays to half-open [start, end) bounds."""
    with np.load(_require_file(path)) as data:
        keys = set(data.files)
        start_key = "start_bins" if "start_bins" in keys else "trial_starts" if "trial_starts" in keys else None
        end_key = "end_bins" if "end_bins" in keys else "trial_ends" if "trial_ends" in keys else None
        if start_key is None or end_key is None:
            raise ValueError(f"Trial info file {path} missing start/end bins. Found keys: {sorted(keys)}")
        starts = np.asarray(data[start_key], dtype=np.int64)
        ends_raw = np.asarray(data[end_key], dtype=np.int64)

    if starts.ndim != 1 or ends_raw.ndim != 1 or starts.shape != ends_raw.shape:
        raise ValueError(f"Invalid trial arrays in {path}: starts {starts.shape}, ends {ends_raw.shape}")

    ends = normalize_trial_ends(starts, ends_raw, n_bins=n_bins)
    return starts, ends



def normalize_trial_ends(starts: np.ndarray, ends_raw: np.ndarray, n_bins: int | None) -> np.ndarray:
    """Normalize trial end bins to exclusive bounds.

    Heuristic:
    - If any end equals `n_bins`, interpret as already exclusive.
    - Else if all ends are < `n_bins`, treat as inclusive and add 1.
    - Validate positive widths and monotonic bounds.
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends_raw = np.asarray(ends_raw, dtype=np.int64)
    if starts.size == 0:
        return ends_raw.copy()
    if np.any(starts < 0):
        raise ValueError("Negative trial start bin detected")

    if n_bins is not None and np.any(starts >= n_bins):
        raise ValueError("Trial start bins exceed session length")

    if n_bins is not None and np.any(ends_raw > n_bins):
        # Likely inclusive `n_bins-1` max is valid, > n_bins is invalid either way.
        raise ValueError("Trial end bins exceed session length")

    if n_bins is not None and np.any(ends_raw == n_bins):
        ends = ends_raw.copy()
    elif n_bins is not None and np.all(ends_raw < n_bins):
        ends = ends_raw + 1
    else:
        # Fallback when n_bins unknown: infer by width positivity under both conventions.
        exclusive_ok = np.all(ends_raw > starts)
        inclusive_ok = np.all((ends_raw + 1) > starts)
        if exclusive_ok and not inclusive_ok:
            ends = ends_raw.copy()
        elif inclusive_ok and not exclusive_ok:
            ends = ends_raw + 1
        else:
            ends = ends_raw + 1

    if np.any(ends <= starts):
        raise ValueError("Non-positive trial width after end-bin normalization")
    if np.any(starts[1:] < ends[:-1]):
        LOGGER.debug("Trials overlap after normalization; proceeding because dataset may allow adjacency/overlap")
    if n_bins is not None and np.any(ends > n_bins):
        raise ValueError("Normalized trial ends exceed session length")
    return ends.astype(np.int64)



def _session_file_triplet(base_dir: Path, session_id: str, is_test: bool) -> tuple[Path, Path, Path, Path]:
    if is_test:
        sbp_path = base_dir / f"{session_id}_sbp_masked.npy"
        mask_path = base_dir / f"{session_id}_mask.npy"
    else:
        sbp_path = base_dir / f"{session_id}_sbp.npy"
        mask_path = base_dir / f"{session_id}_mask.npy"  # not used for train; may not exist
    kin_path = base_dir / f"{session_id}_kinematics.npy"
    trial_info_path = base_dir / f"{session_id}_trial_info.npz"
    return sbp_path, mask_path, kin_path, trial_info_path



def load_train_session(session_id: str, config: Config) -> dict[str, Any]:
    """Load a training session.

    Returns a dictionary containing SBP, kinematics, trial bounds and metadata.
    """
    sbp_path, _, kin_path, trial_info_path = _session_file_triplet(config.train_dir, session_id, is_test=False)
    sbp = np.load(_require_file(sbp_path)).astype(np.float32)
    kin = np.load(_require_file(kin_path)).astype(np.float32)
    starts, ends = _load_trial_info(trial_info_path, n_bins=sbp.shape[0])
    _validate_session_shapes(sbp, kin, starts, ends, config.expected_n_channels)
    return {
        "session_id": session_id,
        "split": "train",
        "sbp": sbp,
        "kinematics": kin,
        "trial_starts": starts,
        "trial_ends": ends,
        "n_trials": int(starts.shape[0]),
        "day": get_session_day(session_id, config),
    }



def load_test_session(session_id: str, config: Config) -> dict[str, Any]:
    """Load a test session with masked SBP and corresponding boolean mask."""
    sbp_path, mask_path, kin_path, trial_info_path = _session_file_triplet(config.test_dir, session_id, is_test=True)
    sbp_masked = np.load(_require_file(sbp_path)).astype(np.float32)
    mask = np.load(_require_file(mask_path)).astype(bool)
    kin = np.load(_require_file(kin_path)).astype(np.float32)
    starts, ends = _load_trial_info(trial_info_path, n_bins=sbp_masked.shape[0])
    _validate_session_shapes(sbp_masked, kin, starts, ends, config.expected_n_channels)
    if mask.shape != sbp_masked.shape:
        raise ValueError(f"Mask shape mismatch for {session_id}: {mask.shape} vs {sbp_masked.shape}")

    masked_trials = identify_masked_trials(mask, starts, ends)
    return {
        "session_id": session_id,
        "split": "test",
        "sbp_masked": sbp_masked,
        "mask": mask,
        "kinematics": kin,
        "trial_starts": starts,
        "trial_ends": ends,
        "n_trials": int(starts.shape[0]),
        "masked_trials": masked_trials,
        "day": get_session_day(session_id, config),
    }



def _validate_session_shapes(
    sbp_like: np.ndarray,
    kinematics: np.ndarray,
    trial_starts: np.ndarray,
    trial_ends: np.ndarray,
    expected_n_channels: int,
) -> None:
    """Validate per-session array shapes and types."""
    if sbp_like.ndim != 2 or sbp_like.shape[1] != expected_n_channels:
        raise ValueError(f"SBP array must have shape (N, {expected_n_channels}), got {sbp_like.shape}")
    if kinematics.ndim != 2 or kinematics.shape[1] != 4 or kinematics.shape[0] != sbp_like.shape[0]:
        raise ValueError(f"Kinematics shape mismatch: {kinematics.shape} for SBP {sbp_like.shape}")
    if trial_starts.ndim != 1 or trial_ends.ndim != 1 or trial_starts.shape != trial_ends.shape:
        raise ValueError("Invalid trial start/end arrays")
    if trial_starts.size == 0:
        raise ValueError("Session has zero trials")



def zscore_session(
    sbp: np.ndarray,
    return_params: bool = True,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """Per-channel z-score normalization.

    Args:
        sbp: Array of shape (N, C).
        return_params: Whether to return means and stds.
        eps: Minimum std clamp.

    Returns:
        Either normalized array or `(normalized, means, stds)`.
    """
    sbp = np.asarray(sbp, dtype=np.float32)
    if sbp.ndim != 2:
        raise ValueError(f"Expected 2D SBP array, got {sbp.shape}")
    means = sbp.mean(axis=0, keepdims=True).astype(np.float32)
    stds = sbp.std(axis=0, keepdims=True).astype(np.float32)
    stds = np.clip(stds, eps, None)
    normed = (sbp - means) / stds
    if return_params:
        return normed.astype(np.float32), means.squeeze(0), stds.squeeze(0)
    return normed.astype(np.float32)



def unzscore(sbp_normed: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Reverse per-channel z-score normalization."""
    sbp_normed = np.asarray(sbp_normed, dtype=np.float32)
    means = np.asarray(means, dtype=np.float32).reshape(1, -1)
    stds = np.asarray(stds, dtype=np.float32).reshape(1, -1)
    if sbp_normed.ndim == 1:
        return (sbp_normed.reshape(1, -1) * stds + means).astype(np.float32).squeeze(0)
    return (sbp_normed * stds + means).astype(np.float32)



def identify_masked_trials(mask: np.ndarray, trial_starts: np.ndarray, trial_ends: np.ndarray) -> list[tuple[int, int]]:
    """Identify trial intervals containing any masked entries.

    Args:
        mask: Boolean array of shape (N, C), True indicates masked entries.
        trial_starts: Trial start bins.
        trial_ends: Trial end bins (exclusive).

    Returns:
        List of `(start_bin, end_bin)` tuples for trials containing at least one masked entry.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {mask.shape}")
    masked_trials: list[tuple[int, int]] = []
    for start, end in zip(trial_starts.tolist(), trial_ends.tolist()):
        if bool(mask[start:end].any()):
            masked_trials.append((int(start), int(end)))
    return masked_trials



def _masked_trial_indices(mask: np.ndarray, trial_starts: np.ndarray, trial_ends: np.ndarray) -> list[int]:
    return [i for i, (s, e) in enumerate(zip(trial_starts.tolist(), trial_ends.tolist())) if bool(mask[s:e].any())]



def extract_unmasked_trials(session_dict: dict[str, Any]) -> np.ndarray:
    """Return fully observed SBP bins from unmasked trials in a test session.

    For test sessions, this removes the 10 held-out masked trials. For train sessions, it returns
    the original SBP array.
    """
    if session_dict.get("split") == "train":
        return np.asarray(session_dict["sbp"], dtype=np.float32)

    sbp_masked = np.asarray(session_dict["sbp_masked"], dtype=np.float32)
    mask = np.asarray(session_dict["mask"], dtype=bool)
    starts = np.asarray(session_dict["trial_starts"], dtype=np.int64)
    ends = np.asarray(session_dict["trial_ends"], dtype=np.int64)
    keep_segments: list[np.ndarray] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        if not bool(mask[start:end].any()):
            keep_segments.append(sbp_masked[start:end])
    if not keep_segments:
        raise ValueError("No unmasked trials found in test session")
    return np.concatenate(keep_segments, axis=0).astype(np.float32)



def extract_masked_trials(session_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return masked trial SBP and masks concatenated over the held-out trials."""
    if session_dict.get("split") != "test":
        raise ValueError("extract_masked_trials is only valid for test sessions")
    sbp_masked = np.asarray(session_dict["sbp_masked"], dtype=np.float32)
    mask = np.asarray(session_dict["mask"], dtype=bool)
    starts = np.asarray(session_dict["trial_starts"], dtype=np.int64)
    ends = np.asarray(session_dict["trial_ends"], dtype=np.int64)

    masked_segments: list[np.ndarray] = []
    mask_segments: list[np.ndarray] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        if bool(mask[start:end].any()):
            masked_segments.append(sbp_masked[start:end])
            mask_segments.append(mask[start:end])
    if not masked_segments:
        raise ValueError("No masked trials found in test session")
    return np.concatenate(masked_segments, axis=0).astype(np.float32), np.concatenate(mask_segments, axis=0).astype(bool)



def get_context_target_index(context_bins: int) -> int:
    """Return deterministic target index for temporal windows.

    Odd windows center exactly. Even windows use a left-biased center, e.g. 8 -> 3.
    """
    if context_bins <= 0:
        raise ValueError("context_bins must be positive")
    return context_bins // 2 if context_bins % 2 == 1 else (context_bins // 2 - 1)



def build_temporal_context_windows(sbp: np.ndarray, context_bins: int) -> np.ndarray:
    """Build temporal context windows for each time bin.

    Args:
        sbp: Array of shape (N, C).
        context_bins: Window length.

    Returns:
        Array of shape (N, context_bins, C).
    """
    sbp = np.asarray(sbp, dtype=np.float32)
    if sbp.ndim != 2:
        raise ValueError(f"Expected SBP shape (N, C), got {sbp.shape}")
    n_bins, n_channels = sbp.shape
    if context_bins <= 0:
        raise ValueError("context_bins must be > 0")
    target_idx = get_context_target_index(context_bins)
    left = target_idx
    right = context_bins - 1 - target_idx
    padded = np.pad(sbp, ((left, right), (0, 0)), mode="edge")
    windows = np.empty((n_bins, context_bins, n_channels), dtype=np.float32)
    for i in range(n_bins):
        windows[i] = padded[i : i + context_bins]
    return windows



def create_artificial_mask(
    sbp: np.ndarray,
    n_channels_to_mask: int = 30,
    seed: int | None = None,
    trial_starts: np.ndarray | None = None,
    trial_ends: np.ndarray | None = None,
    constant_within_trial: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create protocol-matching artificial channel masks.

    Args:
        sbp: Fully observed SBP array (N, C).
        n_channels_to_mask: Number of channels masked per time bin.
        seed: Optional RNG seed.
        trial_starts: Optional trial starts.
        trial_ends: Optional trial ends (exclusive).
        constant_within_trial: If True and trial bounds are provided, use one mask pattern per trial.

    Returns:
        Tuple ``(masked_sbp, mask_bool)``.
    """
    sbp = np.asarray(sbp, dtype=np.float32)
    if sbp.ndim != 2:
        raise ValueError(f"Expected SBP shape (N, C), got {sbp.shape}")
    n_bins, n_channels = sbp.shape
    if not (0 < n_channels_to_mask < n_channels):
        raise ValueError(f"Invalid n_channels_to_mask={n_channels_to_mask} for {n_channels} channels")

    rng = np.random.default_rng(seed)
    mask = np.zeros((n_bins, n_channels), dtype=bool)

    if constant_within_trial and trial_starts is not None and trial_ends is not None:
        starts = np.asarray(trial_starts, dtype=np.int64)
        ends = np.asarray(trial_ends, dtype=np.int64)
        if starts.shape != ends.shape:
            raise ValueError("trial_starts and trial_ends shape mismatch")
        for start, end in zip(starts.tolist(), ends.tolist()):
            idx = rng.choice(n_channels, size=n_channels_to_mask, replace=False)
            mask[start:end, idx] = True
    else:
        for t in range(n_bins):
            idx = rng.choice(n_channels, size=n_channels_to_mask, replace=False)
            mask[t, idx] = True

    masked = sbp.copy()
    masked[mask] = 0.0
    return masked.astype(np.float32), mask



def trial_bounds_from_session(session_dict: dict[str, Any]) -> list[tuple[int, int]]:
    """Return a list of half-open trial bounds `(start, end)` from a session dict."""
    starts = np.asarray(session_dict["trial_starts"], dtype=np.int64)
    ends = np.asarray(session_dict["trial_ends"], dtype=np.int64)
    return [(int(s), int(e)) for s, e in zip(starts.tolist(), ends.tolist())]



def assign_trial_indices(time_bins: np.ndarray, trial_starts: np.ndarray, trial_ends: np.ndarray) -> np.ndarray:
    """Assign each absolute time bin to a trial index.

    Args:
        time_bins: 1D array of absolute time bins.
        trial_starts: Trial starts.
        trial_ends: Trial ends (exclusive).

    Returns:
        1D array of trial indices.
    """
    time_bins = np.asarray(time_bins, dtype=np.int64)
    starts = np.asarray(trial_starts, dtype=np.int64)
    ends = np.asarray(trial_ends, dtype=np.int64)
    out = np.full(time_bins.shape, -1, dtype=np.int64)
    for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        sel = (time_bins >= s) & (time_bins < e)
        out[sel] = i
    if np.any(out < 0):
        raise ValueError("Some time bins do not fall within any trial interval")
    return out



def build_prediction_dataframe_from_dense(
    session_id: str,
    dense_predictions: np.ndarray,
    mask_df_session: pd.DataFrame,
) -> pd.DataFrame:
    """Create row-wise prediction DataFrame from a dense `N x C` prediction array.

    Args:
        session_id: Session ID.
        dense_predictions: Dense predictions aligned to absolute time bins.
        mask_df_session: Filtered `test_mask.csv` rows for this session.

    Returns:
        DataFrame with columns `[sample_id, session_id, time_bin, channel, predicted_sbp]`.
    """
    df = mask_df_session[["sample_id", "session_id", "time_bin", "channel"]].copy()
    if df.empty:
        raise ValueError(f"No mask rows for session {session_id}")
    if not np.all(df["session_id"].astype(str).to_numpy() == session_id):
        raise ValueError("mask_df_session contains rows from multiple sessions")
    time_bins = df["time_bin"].to_numpy(dtype=np.int64)
    channels = df["channel"].to_numpy(dtype=np.int64)
    if dense_predictions.ndim != 2:
        raise ValueError(f"dense_predictions must be 2D, got {dense_predictions.shape}")
    if time_bins.max(initial=-1) >= dense_predictions.shape[0] or channels.max(initial=-1) >= dense_predictions.shape[1]:
        raise ValueError("Prediction array is smaller than required by mask rows")
    df["predicted_sbp"] = dense_predictions[time_bins, channels].astype(np.float32)
    return df



def build_pseudo_solution_df(
    session_id: str,
    true_sbp: np.ndarray,
    mask: np.ndarray,
    sample_id_start: int = 0,
) -> pd.DataFrame:
    """Build a `metric.py`-compatible pseudo-solution for artificially masked data.

    Args:
        session_id: Session ID label for grouping.
        true_sbp: Ground-truth dense SBP (N, C).
        mask: Boolean mask (N, C) indicating evaluated entries.
        sample_id_start: Starting sample ID offset.

    Returns:
        DataFrame with columns expected by `metric.score` solution input.
    """
    true_sbp = np.asarray(true_sbp, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if true_sbp.shape != mask.shape:
        raise ValueError("true_sbp and mask shape mismatch")
    rows, cols = np.where(mask)
    sample_ids = np.arange(sample_id_start, sample_id_start + rows.shape[0], dtype=np.int64)
    # Channel variance computed across the full session (or provided array in future extension).
    channel_vars = true_sbp.var(axis=0).astype(np.float32)
    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_sbp": true_sbp[rows, cols].astype(np.float32),
            "session_id": session_id,
            "channel": cols.astype(np.int64),
            "channel_var": channel_vars[cols].astype(np.float32),
            "time_bin": rows.astype(np.int64),
        }
    )
    return df



def build_submission_like_df_from_dense(mask: np.ndarray, dense_predictions: np.ndarray, sample_id_start: int = 0) -> pd.DataFrame:
    """Build a submission-like DataFrame for artificially masked entries from dense predictions."""
    mask = np.asarray(mask, dtype=bool)
    dense_predictions = np.asarray(dense_predictions, dtype=np.float32)
    if mask.shape != dense_predictions.shape:
        raise ValueError("mask and dense_predictions shape mismatch")
    rows, cols = np.where(mask)
    sample_ids = np.arange(sample_id_start, sample_id_start + rows.shape[0], dtype=np.int64)
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "predicted_sbp": dense_predictions[rows, cols].astype(np.float32),
        }
    )



def session_max_observed_sbp(test_session_dict: dict[str, Any]) -> float:
    """Return max observed (unmasked) SBP value in a test session for clipping."""
    sbp_masked = np.asarray(test_session_dict["sbp_masked"], dtype=np.float32)
    mask = np.asarray(test_session_dict["mask"], dtype=bool)
    if (~mask).any():
        return float(sbp_masked[~mask].max())
    return float(np.nanmax(sbp_masked))



def contiguous_segments(indices: np.ndarray) -> list[tuple[int, int]]:
    """Convert sorted integer indices into half-open contiguous spans over position space.

    This helper is used for trial-safe smoothing on row-wise predictions where gaps in `time_bin`
    should not be smoothed across.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            segments.append((start, i))
            start = i
    segments.append((start, len(indices)))
    return segments


__all__ = [
    "assign_trial_indices",
    "build_prediction_dataframe_from_dense",
    "build_pseudo_solution_df",
    "build_submission_like_df_from_dense",
    "build_temporal_context_windows",
    "contiguous_segments",
    "create_artificial_mask",
    "extract_masked_trials",
    "extract_unmasked_trials",
    "get_context_target_index",
    "get_nearest_train_day",
    "get_session_day",
    "get_test_session_ids",
    "get_train_session_ids",
    "identify_masked_trials",
    "load_metadata",
    "load_sample_submission",
    "load_test_mask_csv",
    "load_test_session",
    "load_train_session",
    "normalize_trial_ends",
    "session_max_observed_sbp",
    "trial_bounds_from_session",
    "unzscore",
    "zscore_session",
]
