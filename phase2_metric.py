"""Phase 2 R2 metric helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLS = ("index_pos", "mrp_pos")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch for R2: {y_true.shape} vs {y_pred.shape}")
    denom = float(np.square(y_true - y_true.mean()).sum())
    sse = float(np.square(y_pred - y_true).sum())
    if denom <= eps:
        return 1.0 if sse <= eps else float("-inf")
    return 1.0 - (sse / denom)


def mean_session_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] < 2:
        raise ValueError(f"Expected matching (N, >=2) arrays, got {y_true.shape} vs {y_pred.shape}")
    scores = {
        "index_pos": r2_score(y_true[:, 0], y_pred[:, 0]),
        "mrp_pos": r2_score(y_true[:, 1], y_pred[:, 1]),
    }
    return float(np.mean(list(scores.values()))), scores


def score_submission(solution_df: pd.DataFrame, submission_df: pd.DataFrame) -> float:
    """Average R2 across all `(session, target)` groups."""
    required = {"sample_id", "session_id", *TARGET_COLS}
    missing_sol = required - set(solution_df.columns)
    missing_sub = {"sample_id", *TARGET_COLS} - set(submission_df.columns)
    if missing_sol:
        raise ValueError(f"Solution is missing columns: {sorted(missing_sol)}")
    if missing_sub:
        raise ValueError(f"Submission is missing columns: {sorted(missing_sub)}")

    merged = solution_df[["sample_id", "session_id", *TARGET_COLS]].merge(
        submission_df[["sample_id", *TARGET_COLS]],
        on="sample_id",
        how="left",
        suffixes=("_true", "_pred"),
    )
    if merged[[f"{col}_pred" for col in TARGET_COLS]].isna().any().any():
        raise ValueError("Submission is missing sample_id rows")

    scores: list[float] = []
    for _, group in merged.groupby("session_id", sort=False):
        for col in TARGET_COLS:
            scores.append(
                r2_score(
                    group[f"{col}_true"].to_numpy(dtype=np.float64),
                    group[f"{col}_pred"].to_numpy(dtype=np.float64),
                )
            )
    return float(np.mean(scores)) if scores else float("-inf")


def summarize_session_scores(records: Iterable[dict[str, float | int | str]]) -> pd.DataFrame:
    return pd.DataFrame(list(records)).sort_values("session_id").reset_index(drop=True)


__all__ = [
    "TARGET_COLS",
    "mean_session_r2",
    "r2_score",
    "score_submission",
    "summarize_session_scores",
]
