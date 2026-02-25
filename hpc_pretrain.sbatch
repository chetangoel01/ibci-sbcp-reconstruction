#!/bin/bash
#SBATCH --job-name=ibci-mae-pretrain
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --output=logs/slurm_pretrain_%j.log
#SBATCH --error=logs/slurm_pretrain_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
export PYTHONUNBUFFERED=1

module purge
# Torch cluster uses conda-provided Python; CUDA module may or may not be needed.
module load cuda/12.1 2>/dev/null || echo "cuda/12.1 module not found; relying on conda/PyTorch runtime"

# Torch cluster defaults (override via env if needed).
REPO_DIR="${REPO_DIR:-/scratch/${USER}/ibci}"
NETID="${NETID:-${USER}}"
SCRATCH_ROOT="${SCRATCH:-/scratch/${NETID}}"
IBCI_SCRATCH="${IBCI_SCRATCH:-${SCRATCH_ROOT}/ibci}"
SWEEP="${SWEEP:-0}"
CONDA_BASE="${CONDA_BASE:-/scratch/${USER}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ibci-sbp}"

mkdir -p "${IBCI_SCRATCH}" "${REPO_DIR}/logs"

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "No environment found. Set VENV_PATH or ensure ${CONDA_BASE}/etc/profile.d/conda.sh exists." >&2
  exit 1
fi

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  STAGE_DIR="${SLURM_TMPDIR}/ibci"
  mkdir -p "${STAGE_DIR}"
  rsync -a --delete "${REPO_DIR}/train" "${STAGE_DIR}/"
  rsync -a --delete "${REPO_DIR}/test" "${STAGE_DIR}/"
  rsync -a "${REPO_DIR}/metadata.csv" "${REPO_DIR}/sample_submission.csv" "${REPO_DIR}/test_mask.csv" "${STAGE_DIR}/"
  export SLURM_TMPDIR
fi

cleanup() {
  mkdir -p "${IBCI_SCRATCH}/checkpoints" "${IBCI_SCRATCH}/results" "${IBCI_SCRATCH}/logs"
  rsync -a "${REPO_DIR}/checkpoints/" "${IBCI_SCRATCH}/checkpoints/" 2>/dev/null || true
  rsync -a "${REPO_DIR}/results/" "${IBCI_SCRATCH}/results/" 2>/dev/null || true
  rsync -a "${REPO_DIR}/logs/" "${IBCI_SCRATCH}/logs/" 2>/dev/null || true
}
trap cleanup EXIT

cd "${REPO_DIR}"

CMD=(python3 pillar2_mae_train.py --config hpc)
if [[ "${SWEEP}" == "1" ]]; then
  # For sweeps, request a longer wall time at submission, e.g. --time=12:00:00
  CMD+=(--sweep)
fi
if [[ "${FAST_DEV_RUN:-0}" == "1" ]]; then
  CMD+=(--fast_dev_run)
fi
if [[ -n "${RESUME_PATH:-}" ]]; then
  CMD+=(--resume "${RESUME_PATH}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  CMD+=(--epochs "${EPOCHS}")
fi
if [[ -n "${SEED:-}" ]]; then
  CMD+=(--seed "${SEED}")
fi
if [[ -n "${MAX_STEPS_PER_EPOCH:-}" ]]; then
  CMD+=(--max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}")
fi
if [[ -n "${MAE_LR:-}" ]]; then
  CMD+=(--mae_lr "${MAE_LR}")
fi
if [[ -n "${MAE_WARMUP_EPOCHS:-}" ]]; then
  CMD+=(--mae_warmup_epochs "${MAE_WARMUP_EPOCHS}")
fi
if [[ -n "${MAE_MASK_RATIO:-}" ]]; then
  CMD+=(--mae_mask_ratio "${MAE_MASK_RATIO}")
fi
if [[ -n "${MAE_WEIGHT_DECAY:-}" ]]; then
  CMD+=(--mae_weight_decay "${MAE_WEIGHT_DECAY}")
fi
if [[ -n "${MAE_DROPOUT:-}" ]]; then
  CMD+=(--mae_dropout "${MAE_DROPOUT}")
fi
if [[ -n "${MAE_CONTEXT_BINS:-}" ]]; then
  CMD+=(--mae_context_bins "${MAE_CONTEXT_BINS}")
fi
if [[ -n "${MAE_BATCH_SIZE:-}" ]]; then
  CMD+=(--mae_batch_size "${MAE_BATCH_SIZE}")
fi
if [[ -n "${N_TRIALS:-}" && "${SWEEP}" == "1" ]]; then
  CMD+=(--n_trials "${N_TRIALS}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
