"""Phase 2 configuration: kinematic decoding from SBP neural activity."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Phase2Config:
    """Configuration for Phase 2 kinematic decoding pipeline."""

    profile: str
    repo_root: Path
    data_dir: Path
    train_dir: Path
    test_dir: Path
    output_dir: Path
    checkpoints_dir: Path
    results_dir: Path
    logs_dir: Path
    metadata_path: Path
    sample_sub_path: Path
    test_index_path: Path
    device: str
    log_level: str = "INFO"
    seed: int = 42

    n_channels: int = 96
    n_position_outputs: int = 2
    n_kinematics: int = 4  # index_pos, mrp_pos, index_vel, mrp_vel
    sampling_rate_hz: int = 50

    # --- shared training ---
    model_type: str = "gru"
    context_bins: int = 50  # 1 second at 50 Hz
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 60
    warmup_epochs: int = 5
    lr_min_factor: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 4
    val_sessions: int = 15  # chronologically last N train sessions held out
    velocity_aux_weight: float = 0.0  # auxiliary velocity loss weight (0 = disabled)

    # --- GRU model ---
    gru_d_model: int = 128
    gru_n_layers: int = 3
    gru_dropout: float = 0.2

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = str(value) if isinstance(value, Path) else value
        return out


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def _make_config(
    repo_root: Path,
    data_dir: Path,
    output_dir: Path,
    profile: str,
) -> Phase2Config:
    output_dir = output_dir.resolve()
    return Phase2Config(
        profile=profile,
        repo_root=repo_root.resolve(),
        data_dir=data_dir.resolve(),
        train_dir=(data_dir / "train").resolve(),
        test_dir=(data_dir / "test").resolve(),
        output_dir=output_dir,
        checkpoints_dir=(output_dir / "checkpoints").resolve(),
        results_dir=(output_dir / "results").resolve(),
        logs_dir=(output_dir / "logs").resolve(),
        metadata_path=(data_dir / "metadata.csv").resolve(),
        sample_sub_path=(data_dir / "sample_submission.csv").resolve(),
        test_index_path=(data_dir / "test_index.csv").resolve(),
        device=_detect_device(),
    )


def get_config(profile: str = "local") -> Phase2Config:
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    output_dir = repo_root / "outputs"
    return _make_config(repo_root=repo_root, data_dir=data_dir, output_dir=output_dir, profile="local")


def ensure_output_dirs(config: Phase2Config) -> None:
    for path in [config.output_dir, config.checkpoints_dir, config.results_dir, config.logs_dir]:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
