# iBCI SBP Reconstruction

This repository contains experiments, notes, and utilities for the Kaggle intracortical BCI competition focused on reconstructing masked Spiking Band Power (SBP) values across long-term recording sessions with neural signal drift.

## What Is Here

- `competition_information.md`: Competition overview, task rules, data format, and metric notes
- `README.txt`: Kaggle-provided dataset README (kept as source reference)
- `metric.py`: Kaggle-provided scoring function
- `train/`, `test/`, `metadata.csv`, `test_mask.csv`, `sample_submission.csv`: Local competition data (gitignored)

## Goal

Predict masked SBP entries in test sessions using observed neural activity (and optionally kinematics/trial structure) while generalizing across session drift.

## Phase 2 Starter

This repo now also includes a parallel Phase 2 kinematic decoding starter built on top of the Phase 1 codebase:

- `phase2_config.py`: Phase 2 dataset/output configuration
- `phase2_data.py`: loaders for the `Dxxx` session layout and submission template
- `phase2_metric.py`: session-averaged R2 helpers
- `phase2_models.py`: GRU baseline plus a transformer decoder built from the Phase 1 encoder blocks
- `phase2_train.py`: train/evaluate/build-submission entry point
- `phase2_rrr.py`: reduced-rank regression baseline inspired by the decoding lecture slides

The Phase 2 pipeline is intentionally kept separate from the original Phase 1 reconstruction stack so both can coexist in the same repo.

Example:

```bash
python3 phase2_train.py --data_dir /path/to/phase2_dataset --build_submission
```

If the Phase 2 dataset is not stored in the repo root, pass `--data_dir` or set `PHASE2_DATA_DIR`.

The default Phase 2 starter is `model_type=gru`. To try a transfer-learning path that reuses the
Phase 1 transformer encoder weights, run:

```bash
python3 phase2_train.py \
  --data_dir /path/to/phase2_dataset \
  --model_type transformer \
  --phase1_checkpoint /path/to/mae_pretrained.pt \
  --build_submission
```

The lecture slides also suggest reduced-rank regression as an interpretable cross-session decoder.
That baseline is available via:

```bash
python3 phase2_rrr.py \
  --data_dir /path/to/phase2_dataset \
  --context_bins 11 \
  --rank 2 \
  --build_submission
```

## Notes

- Raw Kaggle data files are intentionally ignored in Git.
- This repo is set up to track code, experiments, and documentation only.
