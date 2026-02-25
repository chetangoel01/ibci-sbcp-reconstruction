"""Adaptive ensemble orchestration and final submission generation."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import metric
from config import (
    Config,
    ensure_output_dirs,
    get_config,
    preflight_validate_submission_indices,
    setup_logging,
)
from data_utils import (
    assign_trial_indices,
    build_pseudo_solution_df,
    build_submission_like_df_from_dense,
    contiguous_segments,
    create_artificial_mask,
    extract_unmasked_trials,
    get_test_session_ids,
    load_sample_submission,
    load_test_mask_csv,
    load_test_session,
    session_max_observed_sbp,
)
from pillar1_gaussian import ConditionalGaussian, run_all_sessions as run_gaussian_all_sessions

LOGGER = logging.getLogger(__name__)


PILLAR_GAUSSIAN = "gaussian"
PILLAR_MAE = "mae"



def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None



def load_or_run_gaussian_predictions(config: Config, overwrite: bool = False) -> pd.DataFrame:
    """Load cached Gaussian predictions or run Pillar 1 to generate them."""
    path = config.results_dir / "gaussian_predictions.csv"
    if path.exists() and not overwrite:
        LOGGER.info("Loading cached Gaussian predictions from %s", path)
        return pd.read_csv(path)
    LOGGER.info("Running Gaussian pillar for all sessions")
    return run_gaussian_all_sessions(config)



def load_or_prepare_mae_predictions(config: Config, run_if_missing: bool = False, overwrite: bool = False) -> pd.DataFrame:
    """Load cached MAE predictions or optionally trigger TTT generation."""
    path = config.results_dir / "mae_predictions.csv"
    # `--overwrite` only makes sense for MAE if we are also allowed to regenerate via `--run_mae_if_missing`.
    # Otherwise, prefer loading the cached consolidated CSV instead of raising.
    if path.exists() and (not overwrite or not run_if_missing):
        LOGGER.info("Loading cached MAE predictions from %s", path)
        return pd.read_csv(path)
    if not run_if_missing:
        raise FileNotFoundError(
            f"MAE predictions not found at {path}. Run pillar2_mae_ttt.py --all or use --run_mae_if_missing."
        )
    LOGGER.info("Running TTT to generate MAE predictions (missing cache)")
    try:
        from pillar2_mae_ttt import run_all_sessions as run_ttt_all_sessions
    except Exception as exc:  # pragma: no cover - import dependency path
        raise RuntimeError("Unable to import pillar2_mae_ttt.run_all_sessions") from exc
    return run_ttt_all_sessions(config=config, overwrite=overwrite)



def validate_prediction_rows(pred_df: pd.DataFrame, test_mask_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate and align prediction rows to `test_mask.csv` by `sample_id`."""
    required = {"sample_id", "predicted_sbp"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"{label} predictions missing columns: {sorted(missing)}")
    if pred_df["sample_id"].duplicated().any():
        raise ValueError(f"{label} predictions contain duplicate sample_id values")
    merged = test_mask_df.merge(pred_df[["sample_id", "predicted_sbp"]], on="sample_id", how="left")
    if merged["predicted_sbp"].isna().any():
        n_missing = int(merged["predicted_sbp"].isna().sum())
        raise ValueError(f"{label} predictions missing {n_missing} sample_ids")
    if not np.isfinite(merged["predicted_sbp"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{label} predictions contain non-finite values")
    return merged



def _compute_inverse_nmse_weights(scores: dict[str, float], eps: float) -> dict[str, float]:
    numerators: dict[str, float] = {}
    for key, nmse in scores.items():
        safe_nmse = float(max(nmse, eps)) if np.isfinite(nmse) else float("inf")
        numerators[key] = 0.0 if not np.isfinite(safe_nmse) else 1.0 / safe_nmse
    denom = sum(numerators.values())
    if denom <= 0:
        return {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(scores.keys())}
    return {k: v / denom for k, v in numerators.items()}



def evaluate_gaussian_nmse_on_artificial_masks(test_session_dict: dict[str, Any], config: Config) -> float:
    """Estimate Gaussian NMSE on artificial masks drawn from unmasked trials for one session."""
    sbp_full = extract_unmasked_trials(test_session_dict)
    trial_starts = None
    trial_ends = None
    # Unmasked trials are concatenated, so trial-consistent masking over the concatenated sequence is still useful.
    # To preserve exact trial structure, derive trial bounds from original session and keep only unmasked trials.
    starts = np.asarray(test_session_dict["trial_starts"], dtype=np.int64)
    ends = np.asarray(test_session_dict["trial_ends"], dtype=np.int64)
    mask = np.asarray(test_session_dict["mask"], dtype=bool)
    kept_bounds: list[tuple[int, int]] = []
    offset = 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        if not bool(mask[s:e].any()):
            length = e - s
            kept_bounds.append((offset, offset + length))
            offset += length
    if kept_bounds:
        trial_starts = np.array([s for s, _ in kept_bounds], dtype=np.int64)
        trial_ends = np.array([e for _, e in kept_bounds], dtype=np.int64)

    nmse_values: list[float] = []
    cg = ConditionalGaussian(solve_eps=config.gaussian_solve_eps).fit(sbp_full)
    for i in range(config.ensemble_n_eval_masks):
        masked_sbp, art_mask = create_artificial_mask(
            sbp_full,
            n_channels_to_mask=config.masked_channels_per_bin,
            seed=config.seed + i,
            trial_starts=trial_starts,
            trial_ends=trial_ends,
            constant_within_trial=True,
        )
        pred_dense = cg.predict(masked_sbp, art_mask)
        sol = build_pseudo_solution_df(test_session_dict["session_id"], sbp_full, art_mask)
        sub = build_submission_like_df_from_dense(art_mask, pred_dense)
        nmse_values.append(metric.score(sol, sub, row_id_column_name="sample_id"))
    return float(np.mean(nmse_values)) if nmse_values else float("inf")



def evaluate_mae_nmse_on_artificial_masks(test_session_dict: dict[str, Any], config: Config) -> float:
    """Estimate MAE NMSE on artificial masks for one session using adapted checkpoints if available.

    This function delegates to `pillar2_mae_ttt` to avoid duplicate model-loading logic.
    If MAE evaluation is unavailable, returns `inf` so the ensemble falls back to Gaussian.
    """
    try:
        from pillar2_mae_ttt import evaluate_session_nmse_on_artificial_masks
    except Exception as exc:  # pragma: no cover - optional path during early development
        LOGGER.warning("MAE artificial-mask evaluation unavailable (%s); falling back to Gaussian weight", exc)
        return float("inf")
    try:
        return float(evaluate_session_nmse_on_artificial_masks(test_session_dict, config))
    except Exception as exc:  # pragma: no cover - runtime fallback path
        LOGGER.warning("MAE artificial-mask evaluation failed for %s: %s", test_session_dict["session_id"], exc)
        return float("inf")



def estimate_session_weights(
    config: Config,
    pillar1_only: bool = False,
    gate_mae_if_worse: bool = False,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Estimate per-session pillar weights using artificial masking on unmasked trials.

    Returns:
        Tuple of `(weights_by_session, summary_df)`.
    """
    session_ids = get_test_session_ids(config)
    records: list[dict[str, Any]] = []
    weights: dict[str, dict[str, float]] = {}

    for i, session_id in enumerate(tqdm(session_ids, desc="Ensemble weights"), start=1):
        LOGGER.info("Evaluating ensemble weights for %s (%d/%d)", session_id, i, len(session_ids))
        test_session = load_test_session(session_id, config)
        gaussian_nmse = evaluate_gaussian_nmse_on_artificial_masks(test_session, config)
        mae_nmse = float("inf") if pillar1_only else evaluate_mae_nmse_on_artificial_masks(test_session, config)
        scores = {PILLAR_GAUSSIAN: gaussian_nmse, PILLAR_MAE: mae_nmse}
        w = _compute_inverse_nmse_weights(scores, eps=config.ensemble_weight_eps)
        if pillar1_only:
            w = {PILLAR_GAUSSIAN: 1.0, PILLAR_MAE: 0.0}
        elif gate_mae_if_worse and (not np.isfinite(mae_nmse) or mae_nmse >= gaussian_nmse):
            w = {PILLAR_GAUSSIAN: 1.0, PILLAR_MAE: 0.0}
        weights[session_id] = w
        records.append(
            {
                "session_id": session_id,
                "gaussian_nmse_est": gaussian_nmse,
                "mae_nmse_est": mae_nmse,
                "w_gaussian": w[PILLAR_GAUSSIAN],
                "w_mae": w[PILLAR_MAE],
            }
        )

    summary = pd.DataFrame(records).sort_values("session_id").reset_index(drop=True)
    out_path = config.results_dir / "ensemble_session_summary.csv"
    summary.to_csv(out_path, index=False)
    with (config.results_dir / "ensemble_weights.json").open("w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    LOGGER.info("Saved ensemble weights to %s", config.results_dir / "ensemble_weights.json")
    return weights, summary


def load_cached_session_weights(config: Config) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Load cached per-session ensemble weights and summary from disk."""
    weights_path = config.results_dir / "ensemble_weights.json"
    summary_path = config.results_dir / "ensemble_session_summary.csv"
    if not weights_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing cached weights artifacts: {weights_path} and/or {summary_path}")

    with weights_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid ensemble weights JSON payload in {weights_path}")

    weights: dict[str, dict[str, float]] = {}
    for session_id, mapping in payload.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"Invalid weight record for session {session_id!r} in {weights_path}")
        weights[str(session_id)] = {
            PILLAR_GAUSSIAN: float(mapping.get(PILLAR_GAUSSIAN, 0.0)),
            PILLAR_MAE: float(mapping.get(PILLAR_MAE, 0.0)),
        }

    summary_df = pd.read_csv(summary_path)
    return weights, summary_df


def apply_mae_gate_from_summary(
    weights_by_session: dict[str, dict[str, float]],
    summary_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Return a copy of weights with MAE zeroed where cached summary shows no proxy win."""
    required = {"session_id", "gaussian_nmse_est", "mae_nmse_est"}
    if summary_df.empty or not required.issubset(set(summary_df.columns)):
        return weights_by_session

    out: dict[str, dict[str, float]] = {
        str(sid): {PILLAR_GAUSSIAN: float(w.get(PILLAR_GAUSSIAN, 0.0)), PILLAR_MAE: float(w.get(PILLAR_MAE, 0.0))}
        for sid, w in weights_by_session.items()
    }
    gated = 0
    for row in summary_df.itertuples(index=False):
        sid = str(getattr(row, "session_id"))
        g_nmse = float(getattr(row, "gaussian_nmse_est"))
        m_nmse = float(getattr(row, "mae_nmse_est"))
        if (not np.isfinite(m_nmse)) or m_nmse >= g_nmse:
            out[sid] = {PILLAR_GAUSSIAN: 1.0, PILLAR_MAE: 0.0}
            gated += 1
    LOGGER.info("Applied cached-summary MAE gate to %d sessions", gated)
    return out



def combine_predictions(
    gaussian_df: pd.DataFrame,
    mae_df: pd.DataFrame | None,
    weights_by_session: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Combine pillar predictions using per-session weights."""
    base = gaussian_df[["sample_id", "session_id", "time_bin", "channel", "predicted_sbp"]].copy()
    base = base.rename(columns={"predicted_sbp": "gaussian_pred"})

    if mae_df is None:
        base["mae_pred"] = np.nan
        base["w_gaussian"] = base["session_id"].map(lambda s: weights_by_session.get(str(s), {}).get(PILLAR_GAUSSIAN, 1.0))
        base["w_mae"] = 0.0
        base["predicted_sbp"] = base["gaussian_pred"]
        return base

    merged = base.merge(
        mae_df[["sample_id", "predicted_sbp"]].rename(columns={"predicted_sbp": "mae_pred"}),
        on="sample_id",
        how="left",
    )
    if merged["mae_pred"].isna().any():
        n_missing = int(merged["mae_pred"].isna().sum())
        raise ValueError(f"MAE predictions missing {n_missing} rows after merge")

    merged["w_gaussian"] = merged["session_id"].map(
        lambda s: float(weights_by_session.get(str(s), {}).get(PILLAR_GAUSSIAN, 0.5))
    )
    merged["w_mae"] = merged["session_id"].map(lambda s: float(weights_by_session.get(str(s), {}).get(PILLAR_MAE, 0.5)))
    merged["predicted_sbp"] = merged["w_gaussian"] * merged["gaussian_pred"] + merged["w_mae"] * merged["mae_pred"]
    return merged



def apply_session_clipping(pred_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Clip predictions to `[0, max_observed_sbp_in_session]` for each session."""
    out = pred_df.copy()
    session_max: dict[str, float] = {}
    for session_id in tqdm(sorted(out["session_id"].astype(str).unique().tolist()), desc="Session clipping"):
        sess = load_test_session(session_id, config)
        session_max[session_id] = session_max_observed_sbp(sess)

    out["session_max"] = out["session_id"].astype(str).map(session_max)
    out["predicted_sbp"] = out["predicted_sbp"].clip(lower=0.0)
    out["predicted_sbp"] = np.minimum(out["predicted_sbp"].to_numpy(dtype=np.float64), out["session_max"].to_numpy(dtype=np.float64))
    return out.drop(columns=["session_max"])



def _smooth_group_in_place(values: np.ndarray, sigma: float, time_bins: np.ndarray) -> np.ndarray:
    if values.size <= 1 or sigma <= 0:
        return values
    smoothed = values.copy()
    for start, end in contiguous_segments(time_bins):
        if end - start <= 1:
            continue
        smoothed[start:end] = gaussian_filter1d(smoothed[start:end], sigma=sigma, mode="nearest")
    return smoothed



def apply_trial_safe_smoothing(pred_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Apply Gaussian smoothing within each `(session, trial, channel)` group only.

    Smoothing never crosses trial boundaries and is further segmented by contiguous `time_bin`
    runs to avoid bleeding across gaps.
    """
    if not config.ensemble_enable_smoothing or config.ensemble_smooth_sigma <= 0:
        return pred_df

    out = pred_df.copy()
    out["trial_index"] = -1

    for session_id in tqdm(sorted(out["session_id"].astype(str).unique().tolist()), desc="Assign trials"):
        sess = load_test_session(session_id, config)
        sel = out["session_id"].astype(str) == session_id
        time_bins = out.loc[sel, "time_bin"].to_numpy(dtype=np.int64)
        trial_idx = assign_trial_indices(time_bins, sess["trial_starts"], sess["trial_ends"])
        out.loc[sel, "trial_index"] = trial_idx

    smoothed_parts: list[pd.DataFrame] = []
    group_cols = ["session_id", "trial_index", "channel"]
    for _, group in tqdm(out.groupby(group_cols, sort=False), desc="Temporal smoothing"):
        g = group.sort_values("time_bin").copy()
        vals = g["predicted_sbp"].to_numpy(dtype=np.float32)
        times = g["time_bin"].to_numpy(dtype=np.int64)
        g["predicted_sbp"] = _smooth_group_in_place(vals, config.ensemble_smooth_sigma, times)
        smoothed_parts.append(g)

    result = pd.concat(smoothed_parts, axis=0, ignore_index=True)
    result = result.drop(columns=["trial_index"]).sort_values("sample_id").reset_index(drop=True)
    return result



def build_submission(pred_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Merge final predictions into the Kaggle sample submission schema."""
    sample = load_sample_submission(config)
    merged = sample.drop(columns=["predicted_sbp"]).merge(
        pred_df[["sample_id", "predicted_sbp"]], on="sample_id", how="left"
    )
    if merged["predicted_sbp"].isna().any():
        n_missing = int(merged["predicted_sbp"].isna().sum())
        raise ValueError(f"Final submission missing {n_missing} predictions")
    if len(merged) != config.expected_test_mask_rows:
        raise ValueError(
            f"Submission row count mismatch: expected {config.expected_test_mask_rows}, got {len(merged)}"
        )
    return merged.sort_values("sample_id").reset_index(drop=True)



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive ensemble and write submission.csv")
    parser.add_argument("--config", choices=["local", "hpc"], default="local")
    parser.add_argument("--run_mae_if_missing", action="store_true", help="Run TTT if MAE predictions are missing")
    parser.add_argument("--disable_smoothing", action="store_true")
    parser.add_argument("--smooth_sigma", type=float, default=None, help="Override temporal smoothing sigma")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pillar1_only", action="store_true", help="Use only Gaussian pillar but still run post-processing")
    parser.add_argument(
        "--use_cached_weights",
        action="store_true",
        help="Reuse results/ensemble_weights.json and results/ensemble_session_summary.csv if present",
    )
    parser.add_argument(
        "--gate_mae_if_worse",
        action="store_true",
        help="Set MAE weight to 0 for sessions where MAE proxy NMSE is not better than Gaussian",
    )
    return parser.parse_args()



def main() -> None:
    """CLI entry point for the ensemble pipeline."""
    args = _parse_args()
    config = get_config(args.config)
    ensure_output_dirs(config)
    if args.disable_smoothing or args.smooth_sigma is not None:
        updates: dict[str, Any] = {}
        if args.disable_smoothing:
            updates["ensemble_enable_smoothing"] = False
        if args.smooth_sigma is not None:
            sigma = float(args.smooth_sigma)
            updates["ensemble_smooth_sigma"] = sigma
            if sigma <= 0:
                updates["ensemble_enable_smoothing"] = False
        config = replace(config, **updates)
    setup_logging(config.log_level, config.logs_dir / "ensemble.log")
    preflight_validate_submission_indices(config)

    test_mask_df = load_test_mask_csv(config)
    gaussian_raw = load_or_run_gaussian_predictions(config, overwrite=args.overwrite)
    gaussian_df = validate_prediction_rows(gaussian_raw, test_mask_df, label="Gaussian")

    mae_df: pd.DataFrame | None = None
    if not args.pillar1_only:
        mae_raw = load_or_prepare_mae_predictions(config, run_if_missing=args.run_mae_if_missing, overwrite=args.overwrite)
        mae_df = validate_prediction_rows(mae_raw, test_mask_df, label="MAE")

    if args.pillar1_only:
        weights_by_session: dict[str, dict[str, float]] = {}
        summary_df = pd.DataFrame()
        LOGGER.info("Skipping ensemble weight estimation in --pillar1_only mode")
    elif args.use_cached_weights:
        try:
            weights_by_session, summary_df = load_cached_session_weights(config)
            LOGGER.info("Loaded cached ensemble weights for %d sessions", len(weights_by_session))
            if args.gate_mae_if_worse:
                weights_by_session = apply_mae_gate_from_summary(weights_by_session, summary_df)
        except Exception as exc:
            LOGGER.warning("Unable to load cached ensemble weights (%s); recomputing", exc)
            weights_by_session, summary_df = estimate_session_weights(
                config, pillar1_only=False, gate_mae_if_worse=args.gate_mae_if_worse
            )
            LOGGER.info("Estimated ensemble weights for %d sessions", len(weights_by_session))
    else:
        weights_by_session, summary_df = estimate_session_weights(
            config, pillar1_only=False, gate_mae_if_worse=args.gate_mae_if_worse
        )
        LOGGER.info("Estimated ensemble weights for %d sessions", len(weights_by_session))

    combined = combine_predictions(gaussian_df, mae_df, weights_by_session)
    combined = apply_session_clipping(combined, config)
    combined = apply_trial_safe_smoothing(combined, config)
    combined = apply_session_clipping(combined, config)

    final_submission = build_submission(combined, config)
    final_path = config.output_dir / "submission.csv"
    final_submission.to_csv(final_path, index=False)
    LOGGER.info("Saved final submission to %s", final_path)

    if not summary_df.empty:
        LOGGER.info(
            "Weight summary (mean w_gaussian=%.3f, mean w_mae=%.3f)",
            float(summary_df["w_gaussian"].mean()),
            float(summary_df["w_mae"].mean()),
        )


if __name__ == "__main__":
    main()
