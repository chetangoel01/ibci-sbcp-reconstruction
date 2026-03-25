#!/bin/bash
# Train the Phase 2 neural decoder on NYU Torch.

#SBATCH --job-name=ibci-phase2-train
#SBATCH --output=logs/slurm_phase2_train_%j.log
#SBATCH --error=logs/slurm_phase2_train_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
export PYTHONUNBUFFERED=1

module purge
module load cuda/12.1 2>/dev/null || echo "cuda/12.1 module not found; relying on conda/PyTorch runtime"

REPO_DIR="${REPO_DIR:-/scratch/${USER}/ibci}"
CONDA_BASE="${CONDA_BASE:-/scratch/${USER}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ibci-sbp}"
PHASE2_DATA_DIR="${PHASE2_DATA_DIR:-${REPO_DIR}/phase2_v2_kaggle_data}"
PHASE2_OUTPUT_DIR="${PHASE2_OUTPUT_DIR:-/scratch/${USER}/ibci_phase2_outputs}"
MODEL_TYPE="${MODEL_TYPE:-transformer}"

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

CMD=(python3 phase2_train.py --config hpc --data_dir "${PHASE2_DATA_DIR}" --output_dir "${PHASE2_OUTPUT_DIR}" --model_type "${MODEL_TYPE}")

if [[ "${BUILD_SUBMISSION:-1}" == "1" ]]; then
  CMD+=(--build_submission)
fi
if [[ "${FAST_DEV_RUN:-0}" == "1" ]]; then
  CMD+=(--fast_dev_run)
fi
if [[ -n "${PHASE1_CHECKPOINT:-}" ]]; then
  CMD+=(--phase1_checkpoint "${PHASE1_CHECKPOINT}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  CMD+=(--epochs "${EPOCHS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  CMD+=(--batch_size "${BATCH_SIZE}")
fi
if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
  CMD+=(--eval_batch_size "${EVAL_BATCH_SIZE}")
fi
if [[ -n "${CONTEXT_BINS:-}" ]]; then
  CMD+=(--context_bins "${CONTEXT_BINS}")
fi
if [[ -n "${HIDDEN_SIZE:-}" ]]; then
  CMD+=(--hidden_size "${HIDDEN_SIZE}")
fi
if [[ -n "${NUM_LAYERS:-}" ]]; then
  CMD+=(--num_layers "${NUM_LAYERS}")
fi
if [[ -n "${TRANSFORMER_D_MODEL:-}" ]]; then
  CMD+=(--transformer_d_model "${TRANSFORMER_D_MODEL}")
fi
if [[ -n "${TRANSFORMER_N_LAYERS:-}" ]]; then
  CMD+=(--transformer_n_layers "${TRANSFORMER_N_LAYERS}")
fi
if [[ -n "${TRANSFORMER_N_HEADS:-}" ]]; then
  CMD+=(--transformer_n_heads "${TRANSFORMER_N_HEADS}")
fi
if [[ -n "${TRANSFORMER_DIM_FF:-}" ]]; then
  CMD+=(--transformer_dim_ff "${TRANSFORMER_DIM_FF}")
fi
if [[ -n "${LR:-}" ]]; then
  CMD+=(--lr "${LR}")
fi
if [[ -n "${WEIGHT_DECAY:-}" ]]; then
  CMD+=(--weight_decay "${WEIGHT_DECAY}")
fi
if [[ -n "${STEPS_PER_EPOCH:-}" ]]; then
  CMD+=(--steps_per_epoch "${STEPS_PER_EPOCH}")
fi
if [[ -n "${VAL_SESSIONS:-}" ]]; then
  CMD+=(--val_sessions "${VAL_SESSIONS}")
fi
if [[ -n "${AUX_VELOCITY_WEIGHT:-}" ]]; then
  CMD+=(--aux_velocity_weight "${AUX_VELOCITY_WEIGHT}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
