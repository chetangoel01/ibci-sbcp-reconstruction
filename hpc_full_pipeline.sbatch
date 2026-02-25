#!/bin/bash
#SBATCH --job-name=ibci-full-pipeline
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --output=logs/slurm_full_pipeline_%j.log
#SBATCH --error=logs/slurm_full_pipeline_%j.log
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
export PYTHONUNBUFFERED=1

module purge
# Torch cluster uses conda-provided Python; no Python module load required.

REPO_DIR="${REPO_DIR:-/scratch/${USER}/ibci}"
CONDA_BASE="${CONDA_BASE:-/scratch/${USER}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ibci-sbp}"
mkdir -p "${REPO_DIR}/logs"

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

# If TTT produced per-session CSVs but the consolidated MAE CSV is missing, build it here.
if [[ "${PILLAR1_ONLY:-0}" != "1" ]]; then
  if [[ "${FORCE_CONSOLIDATE_TTT:-0}" == "1" || ( ! -f "${REPO_DIR}/results/mae_predictions.csv" && -d "${REPO_DIR}/results/ttt_predictions" ) ]]; then
    echo "Consolidating per-session TTT predictions into results/mae_predictions.csv"
    python3 consolidate_ttt_predictions.py --config hpc
  fi
fi

# Assumes MAE TTT predictions are available (or use --run_mae_if_missing for sequential generation).
CMD=(python3 ensemble.py --config hpc)
if [[ "${PILLAR1_ONLY:-0}" == "1" ]]; then
  CMD+=(--pillar1_only)
fi
if [[ "${RUN_MAE_IF_MISSING:-0}" == "1" ]]; then
  CMD+=(--run_mae_if_missing)
fi
if [[ "${USE_CACHED_WEIGHTS:-0}" == "1" ]]; then
  CMD+=(--use_cached_weights)
fi
if [[ "${GATE_MAE_IF_WORSE:-0}" == "1" ]]; then
  CMD+=(--gate_mae_if_worse)
fi
if [[ "${DISABLE_SMOOTHING:-0}" == "1" ]]; then
  CMD+=(--disable_smoothing)
fi
if [[ -n "${SMOOTH_SIGMA:-}" ]]; then
  CMD+=(--smooth_sigma "${SMOOTH_SIGMA}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
