"""
Kaggle custom metric for the Neural Activity Decoding competition.

Computes Normalized Mean Squared Error (NMSE) between predicted and true SBP
values at masked locations, normalized by per-(session, channel) variance.
Lower is better. NMSE = 1.0 means predicting the channel mean.

The solution DataFrame has columns: sample_id, true_sbp, session_id, channel, channel_var
The submission DataFrame has columns: sample_id, predicted_sbp
"""

import pandas as pd
import pandas.api.types


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Normalized Mean Squared Error (NMSE) for masked neural signal prediction.

    For each (session, channel) group, MSE is divided by the channel's variance
    in that session, then averaged across all groups. This ensures all channels
    and sessions contribute equally regardless of signal magnitude.

    NMSE = 1.0 corresponds to predicting the channel mean (trivial baseline).
    NMSE < 1.0 means the model captures meaningful signal structure.
    NMSE = 0.0 means perfect prediction.

    Parameters
    ----------
    solution : pd.DataFrame
        Ground truth with columns [sample_id, true_sbp, session_id, channel, channel_var].
    submission : pd.DataFrame
        Participant predictions with columns [sample_id, predicted_sbp].
    row_id_column_name : str
        Name of the row ID column used for alignment (typically 'sample_id').

    Returns
    -------
    float
        The NMSE score (lower is better).

    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "sample_id"
    >>> sol = pd.DataFrame({
    ...     "sample_id": [0, 1, 2, 3],
    ...     "true_sbp": [1.0, 2.0, 3.0, 4.0],
    ...     "session_id": ["S1", "S1", "S1", "S1"],
    ...     "channel": [0, 0, 1, 1],
    ...     "channel_var": [1.0, 1.0, 2.0, 2.0]
    ... })
    >>> sub = pd.DataFrame({"sample_id": [0, 1, 2, 3], "predicted_sbp": [1.0, 2.0, 3.0, 4.0]})
    >>> score(sol.copy(), sub.copy(), row_id_column_name)
    0.0
    """
    # Validate columns
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Submission missing column: {row_id_column_name}")

    pred_col = "predicted_sbp"
    true_col = "true_sbp"

    if pred_col not in submission.columns:
        raise ParticipantVisibleError(
            f"Submission must have a '{pred_col}' column. "
            f"Found columns: {list(submission.columns)}"
        )

    if not pandas.api.types.is_numeric_dtype(submission[pred_col]):
        raise ParticipantVisibleError(
            f"Column '{pred_col}' must contain numeric values."
        )

    if submission[pred_col].isna().any():
        raise ParticipantVisibleError(
            f"Column '{pred_col}' contains NaN values. All entries must be finite numbers."
        )

    if submission[pred_col].abs().eq(float("inf")).any():
        raise ParticipantVisibleError(
            f"Column '{pred_col}' contains infinite values. All entries must be finite numbers."
        )

    if len(submission) != len(solution):
        raise ParticipantVisibleError(
            f"Submission has {len(submission)} rows but expected {len(solution)} rows."
        )

    # Merge on row ID to align
    merged = solution.merge(submission[[row_id_column_name, pred_col]], on=row_id_column_name, how="left")

    if merged[pred_col].isna().any():
        n_missing = int(merged[pred_col].isna().sum())
        raise ParticipantVisibleError(
            f"Submission is missing predictions for {n_missing} sample IDs."
        )

    # Compute squared errors
    merged["sq_err"] = (merged[true_col] - merged[pred_col]) ** 2

    # Group by (session_id, channel) and compute MSE per group
    grouped = merged.groupby(["session_id", "channel"]).agg(
        mse=("sq_err", "mean"),
        channel_var=("channel_var", "first")
    ).reset_index()

    # Guard against zero-variance channels (shouldn't happen, but be safe)
    min_var = 1e-10
    grouped["channel_var"] = grouped["channel_var"].clip(lower=min_var)

    # NMSE = mean of (MSE / Var) across all (session, channel) groups
    grouped["nmse"] = grouped["mse"] / grouped["channel_var"]
    nmse = float(grouped["nmse"].mean())

    return nmse
