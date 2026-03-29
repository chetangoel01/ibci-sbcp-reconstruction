"""Global configuration, reproducibility, and preflight validation utilities."""

from __future__ import annotations

import csv
import logging
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    """Configuration container for the iBCI SBP reconstruction pipeline."""

    profile: str
    repo_root: Path
    data_dir: Path
    train_dir: Path
    test_dir: Path
    output_dir: Path
    checkpoints_dir: Path
    results_dir: Path
    logs_dir: Path
    plots_dir: Path
    ttt_predictions_dir: Path
    optuna_dir: Path
    metadata_path: Path
    sample_sub_path: Path
    test_mask_path: Path
    sweep_best_params_path: Path
    device: str
    log_level: str = "INFO"
    seed: int = 42

    expected_test_mask_rows: int = 468_720
    expected_n_test_sessions: int = 24
    expected_n_channels: int = 96

    shrinkage_method: str = "ledoit_wolf"
    gaussian_solve_eps: float = 1e-6

    mae_d_model: int = 128
    mae_n_encoder_layers: int = 6
    mae_n_decoder_layers: int = 2
    mae_n_heads: int = 8
    mae_dim_ff: int = 512
    mae_dropout: float = 0.1
    mae_mask_ratio: float = 0.3125
    mae_context_bins: int = 8
    mae_batch_size: int = 256
    mae_lr: float = 1e-4
    mae_weight_decay: float = 0.01
    mae_epochs: int = 100
    mae_warmup_epochs: int = 10
    mae_lr_min_factor: float = 0.1
    mae_steps_per_epoch: int = 200
    mae_val_steps: int = 50
    mae_num_workers: int = 0
    mae_grad_clip: float = 1.0
    mae_strict_param_budget: bool = False
    mae_max_visible_tokens_warn: int = 768
    mae_max_visible_tokens_hard: int = 1536

    ttt_lr: float = 1e-3
    ttt_epochs: int = 15
    ttt_momentum: float = 0.9
    ttt_weight_decay: float = 0.15
    ttt_batch_size: int = 256
    ttt_steps_per_epoch: int = 100
    ttt_finetune_mode: str = "full"
    ttt_grad_clip: float = 1.0

    ensemble_n_eval_masks: int = 5
    ensemble_enable_smoothing: bool = True
    ensemble_smooth_sigma: float = 1.0
    ensemble_weight_eps: float = 1e-8

    hpc_netid: str | None = None
    hpc_scratch_root: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable-ish dictionary for logging/checkpoint metadata."""
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = str(value) if isinstance(value, Path) else value
        return out

    @property
    def masked_channels_per_bin(self) -> int:
        """Return the number of channels masked per time bin implied by `mae_mask_ratio`."""
        n_masked = int(round(self.expected_n_channels * self.mae_mask_ratio))
        if n_masked <= 0 or n_masked >= self.expected_n_channels:
            raise ValueError(f"Invalid mask ratio {self.mae_mask_ratio}; got {n_masked} masked channels")
        return n_masked

    @property
    def visible_channels_per_bin(self) -> int:
        return self.expected_n_channels - self.masked_channels_per_bin

    @property
    def visible_tokens_per_sample(self) -> int:
        return self.mae_context_bins * self.visible_channels_per_bin



def _detect_device() -> str:
    """Best-effort device detection without hard dependency on torch at import time."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"



def _make_config(repo_root: Path, data_dir: Path, output_dir: Path, profile: str, netid: str | None = None) -> Config:
    """Construct a config object from path roots."""
    output_dir = output_dir.resolve()
    return Config(
        profile=profile,
        repo_root=repo_root.resolve(),
        data_dir=data_dir.resolve(),
        train_dir=(data_dir / "train").resolve(),
        test_dir=(data_dir / "test").resolve(),
        output_dir=output_dir,
        checkpoints_dir=(output_dir / "checkpoints").resolve(),
        results_dir=(output_dir / "results").resolve(),
        logs_dir=(output_dir / "logs").resolve(),
        plots_dir=(output_dir / "results" / "plots").resolve(),
        ttt_predictions_dir=(output_dir / "results" / "ttt_predictions").resolve(),
        optuna_dir=(output_dir / "optuna").resolve(),
        metadata_path=(data_dir / "metadata.csv").resolve(),
        sample_sub_path=(data_dir / "sample_submission.csv").resolve(),
        test_mask_path=(data_dir / "test_mask.csv").resolve(),
        sweep_best_params_path=(output_dir / "results" / "sweep_best_params.json").resolve(),
        device=_detect_device(),
        hpc_netid=netid,
        hpc_scratch_root=(Path(f"/scratch/{netid}/ibci") if netid else None),
    )



def get_config(profile: str = "local") -> Config:
    """Return a local or HPC configuration profile.

    Args:
        profile: Either ``"local"`` or ``"hpc"``.

    Returns:
        Config: Fully populated configuration object.
    """
    if profile == "hpc":
        return get_hpc_config()
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    return _make_config(repo_root=repo_root, data_dir=data_dir, output_dir=repo_root, profile="local")



def get_hpc_config(netid: str | None = None) -> Config:
    """Return an NYU Greene-friendly configuration.

    Paths are resolved in this order for data and outputs:
    1. ``SLURM_TMPDIR/ibci`` (if staged data exists)
    2. ``/scratch/<netid>/ibci``
    3. Local repo root as fallback
    """
    repo_root = Path(__file__).resolve().parent
    netid = netid or os.environ.get("NETID") or os.environ.get("USER")
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    scratch_root = Path(f"/scratch/{netid}/ibci") if netid else None

    staged_data_dir: Path | None = None
    if slurm_tmpdir:
        candidate = Path(slurm_tmpdir) / "ibci"
        if candidate.exists():
            staged_data_dir = candidate

    data_dir = staged_data_dir or (scratch_root if scratch_root and scratch_root.exists() else repo_root)
    output_dir = scratch_root if scratch_root else repo_root

    cfg = _make_config(repo_root=repo_root, data_dir=data_dir, output_dir=output_dir, profile="hpc", netid=netid)
    if scratch_root:
        cfg = replace(cfg, hpc_scratch_root=scratch_root.resolve())
    return cfg



def load_sweep_overrides(config: Config, path: str | Path | None = None) -> Config:
    """Load best hyperparameters from JSON and apply them to a config.

    Unknown keys are ignored to keep backward compatibility with older sweep files.
    """
    import json

    params_path = Path(path) if path is not None else config.sweep_best_params_path
    if not params_path.exists():
        return config

    with params_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Sweep params file is not a JSON object: {params_path}")

    field_map = set(config.__dataclass_fields__.keys())
    updates: dict[str, Any] = {}
    for key, value in payload.items():
        normalized = key.lower()
        if normalized in field_map:
            updates[normalized] = value
        elif key in field_map:
            updates[key] = value
    return replace(config, **updates) if updates else config



def ensure_output_dirs(config: Config) -> None:
    """Create output directories used across the pipeline."""
    for path in [
        config.output_dir,
        config.checkpoints_dir,
        config.results_dir,
        config.logs_dir,
        config.plots_dir,
        config.ttt_predictions_dir,
        config.optuna_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)



def validate_data_paths(config: Config) -> None:
    """Fail loudly if required data files/directories are missing."""
    missing = [
        p for p in [config.train_dir, config.test_dir, config.metadata_path, config.sample_sub_path, config.test_mask_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required data paths: " + ", ".join(str(p) for p in missing))



def _scan_csv_for_sample_ids(path: Path) -> tuple[int, set[int]]:
    """Return row count and unique integer sample IDs from a CSV file."""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise ValueError(f"CSV missing sample_id column: {path}")
        count = 0
        sample_ids: set[int] = set()
        for row in reader:
            count += 1
            try:
                sample_id = int(row["sample_id"])
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid sample_id in {path}: {row.get('sample_id')!r}") from exc
            sample_ids.add(sample_id)
    return count, sample_ids



def preflight_validate_submission_indices(config: Config) -> None:
    """Validate test/submission index files before expensive model execution.

    Checks row counts and ``sample_id`` uniqueness/alignment for ``test_mask.csv`` and
    ``sample_submission.csv``.
    """
    validate_data_paths(config)
    test_count, test_ids = _scan_csv_for_sample_ids(config.test_mask_path)
    sub_count, sub_ids = _scan_csv_for_sample_ids(config.sample_sub_path)

    if test_count != config.expected_test_mask_rows:
        raise ValueError(
            f"test_mask.csv row count mismatch: expected {config.expected_test_mask_rows}, got {test_count}"
        )
    if sub_count != config.expected_test_mask_rows:
        raise ValueError(
            f"sample_submission.csv row count mismatch: expected {config.expected_test_mask_rows}, got {sub_count}"
        )
    if len(test_ids) != test_count:
        raise ValueError(f"test_mask.csv has duplicate sample_id values ({test_count - len(test_ids)} duplicates)")
    if len(sub_ids) != sub_count:
        raise ValueError(
            f"sample_submission.csv has duplicate sample_id values ({sub_count - len(sub_ids)} duplicates)"
        )
    if test_ids != sub_ids:
        missing_in_sub = len(test_ids - sub_ids)
        missing_in_test = len(sub_ids - test_ids)
        raise ValueError(
            "sample_id mismatch between test_mask.csv and sample_submission.csv: "
            f"missing_in_submission={missing_in_sub}, missing_in_test_mask={missing_in_test}"
        )



def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure root logging and return a module logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("ibci")



def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (if installed)."""
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
        # Determinism flags can reduce performance but improve reproducibility.
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


__all__ = [
    "Config",
    "ensure_output_dirs",
    "get_config",
    "get_hpc_config",
    "load_sweep_overrides",
    "preflight_validate_submission_indices",
    "set_global_seeds",
    "setup_logging",
    "validate_data_paths",
]
