#!/bin/bash
# Run the Phase 2 reduced-rank regression baseline on NYU Torch.

#SBATCH --job-name=ibci-phase2-rrr
#SBATCH --output=logs/slurm_phase2_rrr_%j.log
#SBATCH --error=logs/slurm_phase2_rrr_%j.log
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
export PYTHONUNBUFFERED=1

module purge

REPO_DIR="${REPO_DIR:-/scratch/${USER}/ibci}"
CONDA_BASE="${CONDA_BASE:-/scratch/${USER}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ibci-sbp}"
PHASE2_DATA_DIR="${PHASE2_DATA_DIR:-${REPO_DIR}/phase2_v2_kaggle_data}"
PHASE2_OUTPUT_DIR="${PHASE2_OUTPUT_DIR:-/scratch/${USER}/ibci_phase2_rrr_outputs}"

mkdir -p "${REPO_DIR}/logs" "${PHASE2_OUTPUT_DIR}"

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "No environment found. Set VENV_PATH or ensure ${CONDA_BASE}/etc/profile.d/conda.sh exists." >&2
  exit 1
fi

cd "${REPO_DIR}"

CMD=(python3 phase2_rrr.py --config hpc --data_dir "${PHASE2_DATA_DIR}" --output_dir "${PHASE2_OUTPUT_DIR}")

if [[ "${BUILD_SUBMISSION:-1}" == "1" ]]; then
  CMD+=(--build_submission)
fi
if [[ -n "${CONTEXT_BINS:-}" ]]; then
  CMD+=(--context_bins "${CONTEXT_BINS}")
fi
if [[ -n "${RRR_RANK:-}" ]]; then
  CMD+=(--rank "${RRR_RANK}")
fi
if [[ -n "${RIDGE_ALPHA:-}" ]]; then
  CMD+=(--ridge_alpha "${RIDGE_ALPHA}")
fi
if [[ -n "${MAX_TRAIN_ROWS:-}" ]]; then
  CMD+=(--max_train_rows "${MAX_TRAIN_ROWS}")
fi
if [[ -n "${VAL_SESSIONS:-}" ]]; then
  CMD+=(--val_sessions "${VAL_SESSIONS}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
