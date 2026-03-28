Phase 1 Competition Data — V3 Hybrid
==================================================

Train sessions: 226
Test sessions:  24 (easy: 8, medium: 2, hard: 14)
Total masked entries: 468,720

Session IDs are anonymized (S001, S002, ...) in chronological order.
The 'day' column in metadata.csv gives the relative day from the first session.

Files per training session:
  {session_id}_sbp.npy         — (N, 96) SBP values, float32
  {session_id}_kinematics.npy  — (N, 4)  [index_pos, mrp_pos, index_vel, mrp_vel], float32
  {session_id}_trial_info.npz  — start_bins, end_bins, n_trials

Files per test session:
  {session_id}_sbp_masked.npy  — (N, 96) SBP with masked entries set to 0
  {session_id}_mask.npy        — (N, 96) boolean mask (True = masked)
  {session_id}_kinematics.npy  — (N, 4)  full kinematics (NOT masked)
  {session_id}_trial_info.npz  — start_bins, end_bins, n_trials

CSVs:
  test_mask.csv           — sample_id, session_id, time_bin, channel
  sample_submission.csv   — sample_id, predicted_sbp (all zeros)
  solution.csv            — sample_id, true_sbp (KEEP PRIVATE)
  public_private_split.csv — sample_id, usage (public/private)
  metadata.csv            — all sessions with split/difficulty info

Kinematics columns:
  0: index_pos  — index finger position [0=extended, 1=flexed]
  1: mrp_pos    — middle-ring-pinky position [0, 1]
  2: index_vel  — index finger velocity
  3: mrp_vel    — MRP velocity

Masking:
  Per test session, 10 held-out trials have 30/96 channels randomly masked
  per time bin.  Masked entries in SBP are set to 0.

Metric:
  NMSE = mean[ (predicted - true)^2 / var(channel_c in session_s) ]
  Lower is better.  Channel-mean prediction gives NMSE = 1.0.
