"""Configuration helpers for Phase 2 kinematic decoding."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Phase2Config:
    """Runtime configuration for the Phase 2 decoding pipeline."""

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
    test_index_path: Path | None
    device: str
    log_level: str = "INFO"
    seed: int = 42

    expected_n_channels: int = 96
    expected_kinematic_dims: int = 4
    expected_submission_dims: int = 2

    model_type: str = "gru"
    context_bins: int = 41
    batch_size: int = 128
    eval_batch_size: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    transformer_d_model: int = 128
    transformer_n_layers: int = 6
    transformer_n_heads: int = 8
    transformer_dim_ff: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-3
    epochs: int = 20
    steps_per_epoch: int = 200
    val_sessions: int = 20
    aux_velocity_weight: float = 0.25
    grad_clip: float = 1.0
    clip_positions: bool = True

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = str(value) if isinstance(value, Path) else value
        return out

    @property
    def target_index(self) -> int:
        return self.context_bins // 2 if self.context_bins % 2 == 1 else (self.context_bins // 2 - 1)

    @property
    def input_feature_dim(self) -> int:
        return self.expected_n_channels * 2 + 1


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _default_output_dir(repo_root: Path, profile: str) -> Path:
    if profile == "hpc":
        user = os.environ.get("NETID") or os.environ.get("USER")
        if user:
            return Path(f"/scratch/{user}/ibci_phase2_outputs")
    return repo_root / "phase2_outputs"


def get_phase2_config(
    profile: str = "local",
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Phase2Config:
    """Build a Phase 2 config from local paths or environment overrides."""
    repo_root = Path(__file__).resolve().parent
    resolved_data_dir = Path(
        data_dir
        or os.environ.get("PHASE2_DATA_DIR")
        or repo_root
    ).resolve()
    resolved_output_dir = Path(
        output_dir
        or os.environ.get("PHASE2_OUTPUT_DIR")
        or _default_output_dir(repo_root, profile)
    ).resolve()
    test_index_path = (resolved_data_dir / "test_index.csv").resolve()
    if not test_index_path.exists():
        test_index_path = None

    return Phase2Config(
        profile=profile,
        repo_root=repo_root,
        data_dir=resolved_data_dir,
        train_dir=(resolved_data_dir / "train").resolve(),
        test_dir=(resolved_data_dir / "test").resolve(),
        output_dir=resolved_output_dir,
        checkpoints_dir=(resolved_output_dir / "checkpoints").resolve(),
        results_dir=(resolved_output_dir / "results").resolve(),
        logs_dir=(resolved_output_dir / "logs").resolve(),
        metadata_path=(resolved_data_dir / "metadata.csv").resolve(),
        sample_sub_path=(resolved_data_dir / "sample_submission.csv").resolve(),
        test_index_path=test_index_path,
        device=_detect_device(),
    )


def ensure_phase2_output_dirs(config: Phase2Config) -> None:
    for path in [config.output_dir, config.checkpoints_dir, config.results_dir, config.logs_dir]:
        path.mkdir(parents=True, exist_ok=True)


def validate_phase2_paths(config: Phase2Config) -> None:
    missing = [p for p in [config.train_dir, config.test_dir, config.metadata_path, config.sample_sub_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Phase 2 paths: " + ", ".join(str(p) for p in missing))


__all__ = [
    "Phase2Config",
    "ensure_phase2_output_dirs",
    "get_phase2_config",
    "validate_phase2_paths",
]
