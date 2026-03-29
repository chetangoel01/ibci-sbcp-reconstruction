"""Conditional Gaussian estimator for within-session masked SBP reconstruction."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

from config import Config, ensure_output_dirs, get_config, preflight_validate_submission_indices, setup_logging
from data_utils import (
    build_prediction_dataframe_from_dense,
    extract_unmasked_trials,
    get_test_session_ids,
    load_test_mask_csv,
    load_test_session,
    zscore_session,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ConditionalGaussian:
    """Conditional Gaussian estimator with Ledoit-Wolf covariance shrinkage.

    The model is fit on fully observed SBP from unmasked trials within a single test session,
    and predicts masked channels at each time bin using the linear MMSE conditional mean.
    """

    solve_eps: float = 1e-6
    fitted: bool = False
    mean_raw_: np.ndarray | None = None
    std_raw_: np.ndarray | None = None
    mu_norm_: np.ndarray | None = None
    cov_norm_: np.ndarray | None = None

    def fit(self, sbp_unmasked: np.ndarray) -> "ConditionalGaussian":
        """Fit per-session mean/covariance on fully observed bins.

        Args:
            sbp_unmasked: Array of shape `(N, 96)` from unmasked trials.

        Returns:
            Self.
        """
        sbp_unmasked = np.asarray(sbp_unmasked, dtype=np.float32)
        if sbp_unmasked.ndim != 2:
            raise ValueError(f"sbp_unmasked must be 2D, got {sbp_unmasked.shape}")
        z, means, stds = zscore_session(sbp_unmasked, return_params=True)
        lw = LedoitWolf(store_precision=False, assume_centered=False)
        lw.fit(z)

        self.mean_raw_ = means.astype(np.float32)
        self.std_raw_ = stds.astype(np.float32)
        self.mu_norm_ = np.asarray(getattr(lw, "location_", np.zeros(z.shape[1])), dtype=np.float64)
        self.cov_norm_ = np.asarray(lw.covariance_, dtype=np.float64)
        self.fitted = True

        if self.cov_norm_.shape[0] != sbp_unmasked.shape[1]:
            raise ValueError("Covariance shape mismatch after fitting LedoitWolf")
        return self

    def _check_fitted(self) -> None:
        if not self.fitted or self.mean_raw_ is None or self.std_raw_ is None or self.mu_norm_ is None or self.cov_norm_ is None:
            raise RuntimeError("ConditionalGaussian is not fitted")

    def predict(self, sbp_masked: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Predict masked entries for a session subset.

        Args:
            sbp_masked: Array `(N, 96)` with masked entries set (typically) to zero.
            mask: Boolean array `(N, 96)` with True for masked channels.

        Returns:
            Dense reconstructed array `(N, 96)` in raw SBP space. Observed entries are passed through.
        """
        self._check_fitted()
        assert self.mean_raw_ is not None and self.std_raw_ is not None and self.mu_norm_ is not None and self.cov_norm_ is not None

        x = np.asarray(sbp_masked, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if x.shape != mask.shape:
            raise ValueError(f"sbp_masked and mask shape mismatch: {x.shape} vs {mask.shape}")
        if x.ndim != 2:
            raise ValueError(f"Expected 2D arrays, got {x.shape}")

        n_bins, n_channels = x.shape
        if n_channels != self.mean_raw_.shape[0]:
            raise ValueError(f"Channel count mismatch: model {self.mean_raw_.shape[0]} vs input {n_channels}")

        # Normalize with fit-session statistics.
        x_norm = (x.astype(np.float64) - self.mean_raw_[None, :].astype(np.float64)) / self.std_raw_[None, :].astype(np.float64)
        recon = x_norm.copy()

        if not mask.any():
            return (recon * self.std_raw_[None, :] + self.mean_raw_[None, :]).astype(np.float32)

        unique_patterns, inverse = np.unique(mask, axis=0, return_inverse=True)
        cov = self.cov_norm_
        mu = self.mu_norm_

        for p_idx, pattern in enumerate(unique_patterns):
            rows = np.where(inverse == p_idx)[0]
            if rows.size == 0:
                continue
            masked_idx = np.flatnonzero(pattern)
            if masked_idx.size == 0:
                continue
            observed_idx = np.flatnonzero(~pattern)

            if observed_idx.size == 0:
                recon[np.ix_(rows, masked_idx)] = mu[masked_idx][None, :]
                continue

            sigma_oo = cov[np.ix_(observed_idx, observed_idx)]
            sigma_mo = cov[np.ix_(masked_idx, observed_idx)]
            centered_o = x_norm[np.ix_(rows, observed_idx)] - mu[observed_idx][None, :]

            sigma_oo_reg = sigma_oo + self.solve_eps * np.eye(sigma_oo.shape[0], dtype=np.float64)
            chol = cho_factor(sigma_oo_reg, lower=False, check_finite=False)
            solved = cho_solve(chol, centered_o.T, check_finite=False).T  # (n_rows, n_obs) = centered @ inv(sigma_oo)
            pred_m = mu[masked_idx][None, :] + solved @ sigma_mo.T
            recon[np.ix_(rows, masked_idx)] = pred_m

        recon_raw = recon * self.std_raw_[None, :].astype(np.float64) + self.mean_raw_[None, :].astype(np.float64)
        return recon_raw.astype(np.float32)

    def predict_session(self, test_session_dict: dict[str, Any], config: Config, test_mask_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Run the complete Gaussian pipeline for one test session.

        Args:
            test_session_dict: Output of :func:`data_utils.load_test_session`.
            config: Global config.
            test_mask_df: Optional cached `test_mask.csv`; if omitted it is loaded.

        Returns:
            DataFrame with `[sample_id, session_id, time_bin, channel, predicted_sbp]`.
        """
        session_id = str(test_session_dict["session_id"])
        sbp_unmasked = extract_unmasked_trials(test_session_dict)
        self.fit(sbp_unmasked)

        sbp_masked = np.asarray(test_session_dict["sbp_masked"], dtype=np.float32)
        mask = np.asarray(test_session_dict["mask"], dtype=bool)
        dense_pred = self.predict(sbp_masked, mask)

        mask_rows = (test_mask_df if test_mask_df is not None else load_test_mask_csv(config))
        mask_df_session = mask_rows.loc[mask_rows["session_id"].astype(str) == session_id]
        pred_df = build_prediction_dataframe_from_dense(session_id, dense_pred, mask_df_session)
        if pred_df["predicted_sbp"].isna().any():
            raise ValueError(f"NaN predictions generated for {session_id}")
        return pred_df



def run_all_sessions(config: Config) -> pd.DataFrame:
    """Run Pillar 1 for all test sessions and save predictions to CSV."""
    ensure_output_dirs(config)
    preflight_validate_submission_indices(config)

    test_mask_df = load_test_mask_csv(config)
    session_ids = get_test_session_ids(config)
    outputs: list[pd.DataFrame] = []
    for session_id in tqdm(session_ids, desc="Gaussian sessions"):
        session = load_test_session(session_id, config)
        model = ConditionalGaussian(solve_eps=config.gaussian_solve_eps)
        pred_df = model.predict_session(session, config, test_mask_df=test_mask_df)
        outputs.append(pred_df)

    if not outputs:
        raise ValueError("No Gaussian outputs produced")
    all_preds = pd.concat(outputs, axis=0, ignore_index=True)
    all_preds = all_preds.sort_values("sample_id").reset_index(drop=True)
    expected = test_mask_df[["sample_id", "session_id", "time_bin", "channel"]].sort_values("sample_id").reset_index(drop=True)
    merged = expected.merge(all_preds[["sample_id", "predicted_sbp"]], on="sample_id", how="left")
    if merged["predicted_sbp"].isna().any():
        raise ValueError("Gaussian predictions missing sample IDs")

    out_path = config.results_dir / "gaussian_predictions.csv"
    merged.to_csv(out_path, index=False)
    LOGGER.info("Saved Gaussian predictions to %s", out_path)
    return merged



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conditional Gaussian Pillar 1 predictions")
    parser.add_argument("--config", choices=["local", "hpc"], default="local")
    return parser.parse_args()



def main() -> None:
    """CLI entry point for standalone Gaussian predictions."""
    args = _parse_args()
    config = get_config(args.config)
    ensure_output_dirs(config)
    setup_logging(config.log_level, config.logs_dir / "pillar1_gaussian.log")
    run_all_sessions(config)


if __name__ == "__main__":
    main()
