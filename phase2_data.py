"""Data loading and preprocessing for Phase 2 kinematic decoding."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phase2_config import Phase2Config

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def discover_session_ids(directory: Path) -> list[str]:
    """Return sorted list of session IDs found in a data directory."""
    sids = sorted({f.name.split("_")[0] for f in directory.glob("*_sbp.npy")})
    return sids


# ---------------------------------------------------------------------------
# Per-session loading
# ---------------------------------------------------------------------------

def load_sbp(data_dir: Path, sid: str) -> np.ndarray:
    """Load SBP array (N, 96) for a session."""
    return np.load(data_dir / f"{sid}_sbp.npy").astype(np.float32)


def load_kinematics(data_dir: Path, sid: str) -> np.ndarray:
    """Load kinematics array (N, 4): [index_pos, mrp_pos, index_vel, mrp_vel]."""
    return np.load(data_dir / f"{sid}_kinematics.npy").astype(np.float32)


def load_trial_info(data_dir: Path, sid: str) -> dict:
    """Load trial info npz: start_bins, end_bins, n_trials."""
    npz = np.load(data_dir / f"{sid}_trial_info.npz")
    return {k: npz[k] for k in npz.files}


def get_active_mask(sbp: np.ndarray) -> np.ndarray:
    """Return boolean mask (96,) where True = active channel."""
    return ~(sbp == 0).all(axis=0)


# ---------------------------------------------------------------------------
# Z-score normalization
# ---------------------------------------------------------------------------

def zscore_session(sbp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score SBP per channel using only active channels' statistics.

    Returns (z_sbp, mean, std) where zeroed channels remain zero.
    """
    active = get_active_mask(sbp)
    mean = np.zeros(sbp.shape[1], dtype=np.float32)
    std = np.ones(sbp.shape[1], dtype=np.float32)

    mean[active] = sbp[:, active].mean(axis=0)
    std[active] = sbp[:, active].std(axis=0)
    std[std < 1e-8] = 1.0

    z = np.zeros_like(sbp)
    z[:, active] = (sbp[:, active] - mean[active]) / std[active]
    return z, mean, std


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_train_val(
    session_ids: list[str],
    n_val: int = 15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split session IDs into train and validation sets.

    Uses a fixed random split since Phase 2 doesn't provide chronological day info.
    """
    rng = np.random.RandomState(seed)
    shuffled = list(session_ids)
    rng.shuffle(shuffled)
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------------

class SessionCache:
    """Pre-loads and z-scores all sessions into memory."""

    def __init__(self, data_dir: Path, session_ids: list[str], has_kinematics: bool = True):
        self.session_ids = session_ids
        self.sbp: dict[str, np.ndarray] = {}
        self.kinematics: dict[str, np.ndarray] = {}
        self.active_masks: dict[str, np.ndarray] = {}
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # (mean, std)

        for sid in session_ids:
            raw_sbp = load_sbp(data_dir, sid)
            z_sbp, mean, std = zscore_session(raw_sbp)
            self.sbp[sid] = z_sbp
            self.active_masks[sid] = get_active_mask(raw_sbp)
            self.stats[sid] = (mean, std)

            if has_kinematics:
                self.kinematics[sid] = load_kinematics(data_dir, sid)

        LOGGER.info("Loaded %d sessions (kinematics=%s)", len(session_ids), has_kinematics)


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """Yields fixed-length windows from cached sessions for training.

    Each sample is a dict with:
      - sbp: (context_bins, 96)      z-scored SBP
      - active_mask: (96,)           True for active channels
      - positions: (context_bins, 2) target finger positions
      - velocities: (context_bins, 2) target velocities (optional aux target)
    """

    def __init__(
        self,
        cache: SessionCache,
        session_ids: list[str],
        context_bins: int = 50,
        stride: int = 25,
        samples_per_epoch: int | None = None,
        augment: bool = False,
        gain_range: tuple[float, float] = (0.5, 1.5),
        channel_dropout_range: tuple[float, float] = (0.0, 0.3),
        noise_std: float = 0.05,
    ):
        self.cache = cache
        self.context_bins = context_bins
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.gain_range = gain_range
        self.channel_dropout_range = channel_dropout_range
        self.noise_std = noise_std

        self.windows: list[tuple[str, int]] = []
        for sid in session_ids:
            n_bins = self.cache.sbp[sid].shape[0]
            for start in range(0, n_bins - context_bins + 1, stride):
                self.windows.append((sid, start))

        LOGGER.info(
            "SlidingWindowDataset: %d windows from %d sessions (ctx=%d, stride=%d)",
            len(self.windows), len(session_ids), context_bins, stride,
        )

    def __len__(self) -> int:
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.samples_per_epoch is not None:
            idx = np.random.randint(len(self.windows))

        sid, start = self.windows[idx]
        end = start + self.context_bins

        sbp = self.cache.sbp[sid][start:end].copy()
        active_mask = self.cache.active_masks[sid].copy()
        kin = self.cache.kinematics[sid][start:end]

        if self.augment:
            rng = np.random
            # Per-channel multiplicative gain
            lo, hi = self.gain_range
            gains = rng.uniform(lo, hi, size=(1, sbp.shape[1])).astype(np.float32)
            sbp = sbp * gains

            # Dynamic channel dropout: zero out random fraction of active channels
            drop_lo, drop_hi = self.channel_dropout_range
            drop_rate = rng.uniform(drop_lo, drop_hi)
            n_active = int(active_mask.sum())
            n_drop = int(drop_rate * n_active)
            if n_drop > 0:
                active_indices = np.where(active_mask)[0]
                drop_indices = rng.choice(active_indices, size=n_drop, replace=False)
                sbp[:, drop_indices] = 0.0
                active_mask[drop_indices] = False

            # Additive Gaussian noise
            if self.noise_std > 0:
                noise = rng.normal(0, self.noise_std, size=sbp.shape).astype(np.float32)
                noise[:, ~active_mask] = 0.0
                sbp = sbp + noise

        return {
            "sbp": torch.from_numpy(sbp),
            "active_mask": torch.from_numpy(active_mask),
            "positions": torch.from_numpy(kin[:, :2].copy()),
            "velocities": torch.from_numpy(kin[:, 2:4].copy()),
        }


class InferenceDataset(Dataset):
    """Yields overlapping windows for a single test session.

    Used at inference time to tile predictions across the full session.
    """

    def __init__(
        self,
        sbp: np.ndarray,
        active_mask: np.ndarray,
        context_bins: int = 50,
        stride: int = 25,
    ):
        self.sbp = sbp
        self.active_mask = active_mask
        self.context_bins = context_bins
        self.stride = stride
        self.n_bins = sbp.shape[0]

        self.starts: list[int] = []
        for s in range(0, self.n_bins - context_bins + 1, stride):
            self.starts.append(s)
        if self.starts and (self.starts[-1] + context_bins) < self.n_bins:
            self.starts.append(self.n_bins - context_bins)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict:
        start = self.starts[idx]
        end = start + self.context_bins
        return {
            "sbp": torch.from_numpy(self.sbp[start:end]),
            "active_mask": torch.from_numpy(self.active_mask),
            "start": start,
        }


# ---------------------------------------------------------------------------
# Submission helpers
# ---------------------------------------------------------------------------

def load_sample_submission(config: Phase2Config) -> pd.DataFrame:
    return pd.read_csv(config.sample_sub_path)


def load_test_index(config: Phase2Config) -> pd.DataFrame:
    return pd.read_csv(config.test_index_path)
