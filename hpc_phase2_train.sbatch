#!/bin/bash
#SBATCH --job-name=phase2-decode
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=logs/slurm_phase2_%j.log
#SBATCH --error=logs/slurm_phase2_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --requeue

set -euo pipefail
export PYTHONUNBUFFERED=1

module purge
module load cuda/12.1 2>/dev/null || echo "cuda module not found; relying on conda/PyTorch runtime"

REPO_DIR="${REPO_DIR:-/scratch/${USER}/ibci}"
NETID="${NETID:-${USER}}"
SCRATCH_ROOT="${SCRATCH:-/scratch/${NETID}}"
P2_SCRATCH="${P2_SCRATCH:-${SCRATCH_ROOT}/ibci-phase2}"
CONDA_BASE="${CONDA_BASE:-/scratch/${USER}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ibci-sbp}"
MODEL="${MODEL:-transformer}"
EPOCHS="${EPOCHS:-60}"

mkdir -p "${P2_SCRATCH}" "${REPO_DIR}/logs"

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "No environment found. Set VENV_PATH or ensure conda.sh exists." >&2
  exit 1
fi

# Stage data to SLURM_TMPDIR for fast I/O
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  STAGE_DIR="${SLURM_TMPDIR}/phase2_data"
  mkdir -p "${STAGE_DIR}"
  # Sync phase2 data from scratch or repo
  DATA_SRC="${P2_SCRATCH}/data"
  if [[ ! -d "${DATA_SRC}/train" ]]; then
    DATA_SRC="${REPO_DIR}/phase2_v2_kaggle_data"
  fi
  rsync -a --delete "${DATA_SRC}/train" "${STAGE_DIR}/"
  rsync -a --delete "${DATA_SRC}/test" "${STAGE_DIR}/"
  rsync -a "${DATA_SRC}/sample_submission.csv" "${DATA_SRC}/test_index.csv" "${STAGE_DIR}/" 2>/dev/null || true
  export SLURM_TMPDIR
fi

cleanup() {
  mkdir -p "${P2_SCRATCH}/checkpoints" "${P2_SCRATCH}/results" "${P2_SCRATCH}/logs"
  rsync -a "${REPO_DIR}/phase2_outputs/checkpoints/" "${P2_SCRATCH}/checkpoints/" 2>/dev/null || true
  rsync -a "${REPO_DIR}/phase2_outputs/results/" "${P2_SCRATCH}/results/" 2>/dev/null || true
  rsync -a "${REPO_DIR}/logs/" "${P2_SCRATCH}/logs/" 2>/dev/null || true
}
trap cleanup EXIT

cd "${REPO_DIR}"

# --- Training ---
CMD=(python3 phase2_train.py --model "${MODEL}" --profile hpc --epochs "${EPOCHS}")
if [[ -n "${LR:-}" ]]; then CMD+=(--lr "${LR}"); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then CMD+=(--batch_size "${BATCH_SIZE}"); fi
if [[ -n "${CONTEXT_BINS:-}" ]]; then CMD+=(--context_bins "${CONTEXT_BINS}"); fi
if [[ -n "${SEED:-}" ]]; then CMD+=(--seed "${SEED}"); fi
if [[ -n "${VEL_AUX:-}" ]]; then CMD+=(--velocity_aux "${VEL_AUX}"); fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# --- Inference ---
echo "Running inference..."
python3 phase2_inference.py --model "${MODEL}" --profile hpc

echo "Done. Submission at phase2_outputs/results/submission_${MODEL}.csv"
