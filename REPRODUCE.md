# Reproducing Best Result: Kaggle R² = 0.7006

## Overview

The best submission is an equal-weight ensemble of three GRU models with different context windows and widths, each trained independently on Modal (A10G GPU).

| Model | d_model | ctx_bins | Val R² | Kaggle R² |
|-------|---------|----------|--------|-----------|
| gru_wide | 256 | 400 | 0.7836 | — |
| gru_ctx800 | 128 | 800 | 0.7840 | 0.6705 |
| gru_ctx600 | 128 | 600 | 0.7787 | 0.6658 |
| **Ensemble (equal 1/3)** | | | **0.7944** | **0.7006** |

## Prerequisites

```bash
pip install torch numpy pandas scipy tqdm
```

Data: place the Phase 2 Kaggle dataset in `phase2_v2_kaggle_data/` with `train/` and `test/` subdirectories containing `{session_id}_sbp.npy` and `{session_id}_kinematics.npy` files.

## Step 1: Train the Three Models

Each model trains for 80 epochs (~1-2 hours on A10G). All use seed=44, AdamW, cosine LR with warmup, velocity auxiliary loss (0.1), and channel dropout augmentation.

### Option A: Modal (recommended)

```bash
modal run --detach modal_train_gru_ctx800.py
modal run --detach modal_train_gru_ctx600.py
modal run --detach modal_train_gru_wide.py
```

Download checkpoints when done:

```bash
mkdir -p phase2_outputs/checkpoints/{gru_ctx800,gru_ctx600,gru_wide}
modal volume get phase2-outputs-gru-ctx800 checkpoints/best_gru.pt phase2_outputs/checkpoints/gru_ctx800/best_gru.pt
modal volume get phase2-outputs-gru-ctx600 checkpoints/best_gru.pt phase2_outputs/checkpoints/gru_ctx600/best_gru.pt
modal volume get phase2-outputs-gru-wide checkpoints/best_gru.pt phase2_outputs/checkpoints/gru_wide/best_gru.pt
```

### Option B: Local / HPC

```bash
python phase2_train.py --model gru --context_bins 800 --seed 44 --epochs 80 --d_model 128
python phase2_train.py --model gru --context_bins 600 --seed 44 --epochs 80 --d_model 128
python phase2_train.py --model gru --context_bins 400 --seed 44 --epochs 80 --d_model 256
```

Note: the local training script uses default `gru_d_model=128`. For the wide model (d=256), use the Modal script or modify the config.

## Step 2: Generate Individual Submissions

```bash
python generate_submissions.py
```

This script:
1. Loads each checkpoint, reading model config (d_model, ctx) from the saved config
2. Runs overlap-tiled inference on all 125 test sessions with stride = ctx/2
3. Applies sigma=3 Gaussian smoothing
4. Clips predictions to [0, 1]
5. Produces individual CSVs and ensemble CSVs

## Step 3: Verify Ensemble on Validation Set (optional)

```bash
python validate_ensemble_fast.py
```

Uses the same seed=44 val split (15 held-out sessions) as training. Expected output:

```
gru_wide        val_R2=0.7836
gru_ctx800      val_R2=0.7840
gru_ctx600      val_R2=0.7787

Best ensemble: gru_wide + gru_ctx800 + gru_ctx600 (equal 1/3) → val_R2=0.7944
```

## Step 4: Submit

The final submission file is `submission_ensemble_equal_top3.csv` — equal-weight average of all three models' predictions.

## Key Hyperparameters

All three models share:

| Parameter | Value |
|-----------|-------|
| Architecture | Bidirectional GRU |
| n_layers | 3 |
| dropout | 0.2 |
| lr | 3e-4 |
| weight_decay | 0.01 |
| warmup_epochs | 5 |
| epochs | 80 |
| batch_size | 64 |
| seed | 44 |
| val_sessions | 15 |
| velocity_aux_weight | 0.1 |
| grad_clip | 1.0 |
| smoothing sigma | 3.0 |
| overlap stride | ctx_bins / 2 |
| overlap weighting | equal (not Hann) |

## What Didn't Work

- **Transformer decoder** (val 0.56) — GRU's recurrent bias is better for temporal neural signals
- **POYO-Perceiver** (val 0.42) — channel-level tokenization with small context insufficient
- **SWA** — hurt Kaggle score despite looking good on val
- **Hann window overlap** — worse than equal-weight averaging
- **MAE pretrained features** — 3.5h extraction, val 0.7781 (worse than plain GRU)
- **LINK external data** — session mapping was wrong, kinematics didn't match
- **Optimized ensemble weights** — barely beat equal weights on val, risk of overfitting
