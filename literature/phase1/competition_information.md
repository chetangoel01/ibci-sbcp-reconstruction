# Competition Information

## Overview

Intracortical brain-computer interfaces (iBCIs) record neural activity from arrays of electrodes implanted in the brain. A major challenge for real-world iBCIs is neural signal drift — the statistical properties of recorded signals change over days, weeks, and months due to electrode migration, tissue remodeling, and neural plasticity. Decoders trained on one day's data can degrade significantly when applied to recordings from a different day.

In this competition, you will work with long-term intracortical recordings spanning hundreds of sessions collected over multiple years from a non-human primate. Your task is to predict masked neural activity (Spiking Band Power) in held-out test sessions given partial observations. This requires learning the spatial and temporal structure of neural signals and generalizing across recording sessions that may be separated by days, weeks, or months.

### What You Will Learn

- How neural signals are structured (spatial correlations across channels, temporal dynamics)
- How signal statistics drift over long timescales
- Techniques for building models that generalize across non-stationary distributions

### Competition Timing (as provided)

- Start: 10 hours ago
- Close: 21 days to go

## Description

### The Data

The dataset consists of recordings from a 96-channel intracortical electrode array (two 64-channel Utah arrays) implanted in the motor cortex of a non-human primate. During each recording session, the subject performed a finger-movement task while neural activity was captured.

Each session provides:

- Spiking Band Power (SBP): A 96-channel neural feature extracted at 50 Hz (20 ms bins), representing the power in the spiking frequency band for each electrode. Shape: `(N_timebins, 96)`.
- Finger Kinematics: 4 behavioral variables — index finger position, middle-ring-pinky (MRP) finger position, index velocity, and MRP velocity — all normalized to `[0, 1]`. Shape: `(N_timebins, 4)`.
- Trial Information: Each session contains multiple trials. Trial start/end time bins are provided.

### The Task

In the test sessions, a subset of SBP values have been masked (set to zero). Specifically, for 10 held-out trials per test session, 30 out of 96 channels are randomly zeroed out at each time bin. A boolean mask array and a CSV index file identify exactly which entries are masked.

Your goal: predict the original (unmasked) SBP values at the masked locations.

This is a self-supervised reconstruction task. You are not given any labels — instead, you must learn from the observed (unmasked) neural activity and any patterns in the training data to fill in the missing values. Think of it like a neural signal version of image inpainting or masked language modeling.

The test set includes sessions from across the full recording timeline. Some test sessions are close in time to training data (where signal statistics are similar), while others are far from any training session (where significant signal drift has occurred). Your model must handle both cases.

## Evaluation

Submissions are evaluated by comparing your predicted SBP values to the hidden ground truth at all masked locations across 24 test sessions (~469K total masked entries).

### NMSE (Normalized Mean Squared Error)

NMSE measures prediction accuracy while normalizing for differences in signal magnitude across channels and sessions. This is important because signal amplitudes change over time due to electrode drift.

Formula:

```text
                       1         Σ (predicted_i - true_i)²
NMSE(s, c) = ─────────────── × ─────────────────────────────
              Var(c in s)              n_sc
```

Where:

- `s` = session
- `c` = channel
- `n_sc` = number of masked entries for channel `c` in session `s`
- `Var(c in s)` = variance of channel `c` across the full session `s` (not just masked entries)

Final Score:

- `NMSE = mean of NMSE(s, c)` across all `(session, channel)` groups

This ensures all sessions and channels contribute equally regardless of signal magnitude. Range: `0` and above (lower is better).

### NMSE Interpretation

| NMSE | Meaning |
|---|---|
| 0 | Perfect prediction |
| < 1.0 | Model captures meaningful signal structure |
| 1.0 | Equivalent to predicting the per-channel mean (trivial baseline) |
| > 1.0 | Worse than the trivial baseline |

### Public/Private Split

- Total masked entries: ~469K
- Public leaderboard: ~234K entries (50%)
- Private leaderboard: ~235K entries (50%)

The final ranking uses the private portion. This prevents overfitting to the leaderboard. On the Kaggle leaderboard, a lower NMSE corresponds to a higher rank.

## Submission Format

Submit a CSV file with the following format:

```text
sample_id,predicted_sbp
0,2.34
1,3.12
2,1.87
...
```

- `sample_id`: Integer index matching the `sample_id` column in `test_mask.csv`. Must cover all masked entries.
- `predicted_sbp`: Your predicted SBP value for that masked entry (float).

The file `sample_submission.csv` provides a template with all zeros — you must replace the `predicted_sbp` values with your own.

## Dataset Description

### Local Repo Layout Note

In this local repository, the Kaggle files are currently extracted directly into the repo root (for example: `train/`, `test/`, `metadata.csv`, `sample_submission.csv`, `test_mask.csv`, `metric.py`, `README.txt`).

### File Structure

```text
├── train/
│   ├── {session_id}_sbp.npy              # (N, 96) SBP values
│   ├── {session_id}_kinematics.npy       # (N, 4)  finger kinematics
│   └── {session_id}_trial_info.npz       # trial start/end bins
├── test/
│   ├── {session_id}_sbp_masked.npy       # (N, 96) SBP with masked entries = 0
│   ├── {session_id}_mask.npy             # (N, 96) boolean mask (True = masked)
│   ├── {session_id}_kinematics.npy       # (N, 4)  kinematics (full, not masked)
│   └── {session_id}_trial_info.npz       # trial start/end bins
├── metadata.csv
├── sample_submission.csv
├── test_mask.csv
└── metric.py
```

- 226 training sessions and 24 test sessions are provided.

### `metadata.csv`

| Column | Description |
|---|---|
| `session_id` | Anonymous session identifier (e.g., S001, S002, …) in chronological order |
| `day` | Relative recording day (day 0 = first session) |
| `split` | `train` or `test` |
| `n_bins` | Number of time bins in the session (~22K–34K at 50 Hz) |
| `n_trials` | Number of trials in the session |

### `sample_submission.csv`

Each row is one masked entry you need to predict. The file serves as both a submission template and a reference for what each `sample_id` corresponds to:

| Column | Description |
|---|---|
| `sample_id` | Unique integer identifying this masked entry |
| `session_id` | Which test session this entry belongs to |
| `time_bin` | Which time bin (row index into the NumPy arrays) |
| `channel` | Which SBP channel (column index, 0-indexed) |
| `predicted_sbp` | Your prediction — replace the zeros with your values |

### NumPy Files

Training sessions (`train/`):

- `{sid}_sbp.npy` — `Float32 (N_timebins, 96)`. Full SBP values, one column per electrode channel.
- `{sid}_kinematics.npy` — `Float32 (N_timebins, 4)`. Columns: `[index_pos, mrp_pos, index_vel, mrp_vel]`, all normalized to `[0, 1]`.
- `{sid}_trial_info.npz` — Contains `start_bins`, `end_bins` (arrays of trial boundaries), and `n_trials`.

Test sessions (`test/`):

- `{sid}_sbp_masked.npy` — `Float32 (N_timebins, 96)`. SBP values with masked entries set to `0`.
- `{sid}_mask.npy` — `Boolean (N_timebins, 96)`. `True = masked` (must predict), `False = observed`.
- `{sid}_kinematics.npy` — `Float32 (N_timebins, 4)`. Full kinematics, not masked.
- `{sid}_trial_info.npz` — Same format as training.

### `metric.py`

The NMSE metric used for scoring is included in the dataset. You can use it for local evaluation:

```python
import pandas as pd
from metric import score

solution = pd.read_csv('solution.csv')       # you won't have this — use for local cross-val
submission = pd.read_csv('my_submission.csv')
print(f"NMSE: {score(solution, submission, 'sample_id'):.4f}")
```

## Dataset Metadata (as provided)

- Files: 779 files
- Size: 2.56 GB
- Type: `npy`, `npz`, `csv` + 2 others
- License: MIT

### Included Items (Kaggle page listing, as provided)

- `kaggle_data/` (2 directories, 5 files)
- `test/` (96 files)
- `train/` (678 files)
- `README.txt`
- `metric.py`
- `metadata.csv`
- `sample_submission.csv`
- `test_mask.csv`

### Data Explorer Summary (as provided)

- `kaggle_data`
- `test`
- `train`
- `README.txt`
- `metric.py`
- `metadata.csv`
- `sample_submission.csv`
- `test_mask.csv`

Summary:

- 779 files
- 16 columns
- Download All
