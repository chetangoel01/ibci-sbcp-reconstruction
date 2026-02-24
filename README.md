# iBCI SBP Reconstruction

This repository contains experiments, notes, and utilities for the Kaggle intracortical BCI competition focused on reconstructing masked Spiking Band Power (SBP) values across long-term recording sessions with neural signal drift.

## What Is Here

- `competition_information.md`: Competition overview, task rules, data format, and metric notes
- `README.txt`: Kaggle-provided dataset README (kept as source reference)
- `metric.py`: Kaggle-provided scoring function
- `train/`, `test/`, `metadata.csv`, `test_mask.csv`, `sample_submission.csv`: Local competition data (gitignored)

## Goal

Predict masked SBP entries in test sessions using observed neural activity (and optionally kinematics/trial structure) while generalizing across session drift.

## Notes

- Raw Kaggle data files are intentionally ignored in Git.
- This repo is set up to track code, experiments, and documentation only.
