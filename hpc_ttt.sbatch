#!/bin/bash
#SBATCH --job-name=ibci-mae-ttt
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --output=logs/slurm_ttt_%A_%a.log
#SBATCH --error=logs/slurm_ttt_%A_%a.log
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-23

set -euo pipefail
export PYTHONUNBUFFERED=1

module purge
# Torch cluster uses conda-provided Python; CUDA module may or may not be needed.
module load cuda/12.1 2>/dev/null || echo "cuda/12.1 module not found; relying on conda/PyTorch runtime"

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

CMD=(python3 pillar2_mae_ttt.py --config hpc --session_index "${SLURM_ARRAY_TASK_ID}")
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "${FAST_DEV_RUN:-0}" == "1" ]]; then
  CMD+=(--fast_dev_run)
fi
if [[ -n "${CKPT_PATH:-}" ]]; then
  CMD+=(--checkpoint "${CKPT_PATH}")
fi
if [[ -n "${CONFIG_PROFILE:-}" ]]; then
  # Override default hpc profile only if explicitly requested.
  CMD=(python3 pillar2_mae_ttt.py --config "${CONFIG_PROFILE}" --session_index "${SLURM_ARRAY_TASK_ID}")
  [[ "${OVERWRITE:-0}" == "1" ]] && CMD+=(--overwrite)
  [[ "${FAST_DEV_RUN:-0}" == "1" ]] && CMD+=(--fast_dev_run)
  [[ -n "${CKPT_PATH:-}" ]] && CMD+=(--checkpoint "${CKPT_PATH}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
