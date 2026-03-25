#!/bin/bash
# Submit a Phase 2 job on NYU Torch.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/${USER}/ibci}"
PHASE2_DATA_DIR="${PHASE2_DATA_DIR:-${REPO_DIR}/phase2_v2_kaggle_data}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
MODE="train"
MODEL_TYPE="${MODEL_TYPE:-transformer}"

usage() {
  cat <<'EOF'
Usage: hpc_phase2_submit.sh [options]

Options:
  --repo-dir PATH          Remote repo root (default: /scratch/$USER/ibci)
  --data-dir PATH          Phase 2 dataset root on HPC
  --account NAME           Optional SLURM account override
  --partition NAME         Optional SLURM partition override
  --qos NAME               Optional SLURM QoS override
  --rrr                    Submit the RRR baseline instead of neural training
  --gru                    Submit the GRU neural model
  --transformer            Submit the transformer neural model (default)
  --fast-dev               Pass FAST_DEV_RUN=1
  --phase1-checkpoint P    Optional Phase 1 checkpoint for transformer initialization
  --epochs N               Override epochs
  --context-bins N         Override context bins
  --batch-size N           Override batch size
  -h, --help               Show help
EOF
}

SBATCH_ARGS=(sbatch)
EXPORTS=("ALL")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --data-dir)
      PHASE2_DATA_DIR="$2"; shift 2 ;;
    --account)
      ACCOUNT="$2"; shift 2 ;;
    --partition)
      PARTITION="$2"; shift 2 ;;
    --qos)
      QOS="$2"; shift 2 ;;
    --rrr)
      MODE="rrr"; shift ;;
    --gru)
      MODE="train"; MODEL_TYPE="gru"; shift ;;
    --transformer)
      MODE="train"; MODEL_TYPE="transformer"; shift ;;
    --fast-dev)
      EXPORTS+=("FAST_DEV_RUN=1"); shift ;;
    --phase1-checkpoint)
      EXPORTS+=("PHASE1_CHECKPOINT=$2"); shift 2 ;;
    --epochs)
      EXPORTS+=("EPOCHS=$2"); shift 2 ;;
    --context-bins)
      EXPORTS+=("CONTEXT_BINS=$2"); shift 2 ;;
    --batch-size)
      EXPORTS+=("BATCH_SIZE=$2"); shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

cd "${REPO_DIR}"
mkdir -p logs

if [[ -n "${PARTITION}" ]]; then
  SBATCH_ARGS+=(--partition "${PARTITION}")
fi
if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_ARGS+=(--account "${ACCOUNT}")
fi
if [[ -n "${QOS}" ]]; then
  SBATCH_ARGS+=(--qos "${QOS}")
fi

EXPORTS+=("REPO_DIR=${REPO_DIR}")
EXPORTS+=("PHASE2_DATA_DIR=${PHASE2_DATA_DIR}")

if [[ "${MODE}" == "rrr" ]]; then
  JOB_SCRIPT="hpc_phase2_rrr.sbatch"
else
  JOB_SCRIPT="hpc_phase2_train.sbatch"
  EXPORTS+=("MODEL_TYPE=${MODEL_TYPE}")
fi

jobid=$("${SBATCH_ARGS[@]}" --parsable --export "$(IFS=,; echo "${EXPORTS[*]}")" "${JOB_SCRIPT}")
jobid=${jobid%%;*}

echo "Submitted ${JOB_SCRIPT} as job ${jobid}"
echo "Monitor: squeue -u $USER"
