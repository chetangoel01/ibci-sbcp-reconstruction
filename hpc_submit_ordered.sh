#!/bin/bash
# Submit the iBCI pipeline in the correct order using SLURM dependencies.
# Default sequence: pretrain -> TTT array -> full pipeline (auto-consolidates TTT CSVs if needed).

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/${USER}/ibci}"
PARTITION="${PARTITION:-}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"
FAST_DEV_RUN="${FAST_DEV_RUN:-0}"
PILLAR1_ONLY="${PILLAR1_ONLY:-0}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
SKIP_TTT="${SKIP_TTT:-0}"
SKIP_ENSEMBLE="${SKIP_ENSEMBLE:-0}"
OVERWRITE_TTT="${OVERWRITE_TTT:-1}"
OVERWRITE_ENSEMBLE="${OVERWRITE_ENSEMBLE:-0}"
FORCE_CONSOLIDATE_TTT="${FORCE_CONSOLIDATE_TTT:-0}"

PRETRAIN_JOBID=""
TTT_JOBID=""
ENSEMBLE_JOBID=""
LAST_DEP=""

usage() {
  cat <<'EOF'
Usage: hpc_submit_ordered.sh [options]

Options:
  --repo-dir PATH            Repo root (default: /scratch/$USER/ibci)
  --partition NAME           Optional SLURM partition override (e.g., h200_tandon)
  --array-concurrency N      Throttle TTT array concurrency (default: 8)
  --fast-dev                 Submit smoke-test-sized jobs (FAST_DEV_RUN=1)
  --pillar1-only             Submit only final CPU job in pillar1-only mode
  --skip-pretrain            Skip pretrain submission (assume pretrained checkpoint exists)
  --skip-ttt                 Skip TTT submission (assume TTT outputs exist)
  --skip-ensemble            Skip final ensemble submission
  --no-overwrite-ttt         Do not pass OVERWRITE=1 to TTT array jobs
  --no-overwrite-ensemble    Do not pass OVERWRITE=1 to ensemble job (default)
  --force-consolidate-ttt    Force consolidation of results/ttt_predictions/*.csv before ensemble
  -h, --help                 Show help

Examples:
  ./hpc_submit_ordered.sh
  ./hpc_submit_ordered.sh --partition h200_tandon --array-concurrency 6
  ./hpc_submit_ordered.sh --skip-pretrain
  ./hpc_submit_ordered.sh --pillar1-only
  ./hpc_submit_ordered.sh --fast-dev --partition h200_tandon
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --partition)
      PARTITION="$2"; shift 2 ;;
    --array-concurrency)
      ARRAY_CONCURRENCY="$2"; shift 2 ;;
    --fast-dev)
      FAST_DEV_RUN=1; shift ;;
    --pillar1-only)
      PILLAR1_ONLY=1; shift ;;
    --skip-pretrain)
      SKIP_PRETRAIN=1; shift ;;
    --skip-ttt)
      SKIP_TTT=1; shift ;;
    --skip-ensemble)
      SKIP_ENSEMBLE=1; shift ;;
    --no-overwrite-ttt)
      OVERWRITE_TTT=0; shift ;;
    --no-overwrite-ensemble)
      OVERWRITE_ENSEMBLE=0; shift ;;
    --force-consolidate-ttt)
      FORCE_CONSOLIDATE_TTT=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repo dir not found: $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"
mkdir -p logs

COMMON_SBAT=(sbatch)
if [[ -n "$PARTITION" ]]; then
  COMMON_SBAT+=(--partition "$PARTITION")
fi

submit_job() {
  local __outvar="$1"
  shift
  local jobid
  jobid=$("${COMMON_SBAT[@]}" --parsable "$@")
  jobid=${jobid%%;*}
  printf -v "$__outvar" '%s' "$jobid"
}

# Pillar1-only implies no GPU stages.
if [[ "$PILLAR1_ONLY" -eq 1 ]]; then
  SKIP_PRETRAIN=1
  SKIP_TTT=1
fi

if [[ "$SKIP_PRETRAIN" -eq 0 ]]; then
  PRE_EXPORT="ALL"
  if [[ "$FAST_DEV_RUN" -eq 1 ]]; then
    PRE_EXPORT+",FAST_DEV_RUN=1,EPOCHS=1,MAX_STEPS_PER_EPOCH=2"
  fi
  echo "Submitting pretrain job..."
  submit_job PRETRAIN_JOBID --export="$PRE_EXPORT" hpc_pretrain.sbatch
  LAST_DEP="$PRETRAIN_JOBID"
  echo "  pretrain job id: $PRETRAIN_JOBID"
else
  echo "Skipping pretrain submission"
fi

if [[ "$SKIP_TTT" -eq 0 ]]; then
  TTT_EXPORT="ALL"
  if [[ "$OVERWRITE_TTT" -eq 1 ]]; then
    TTT_EXPORT+=",OVERWRITE=1"
  fi
  if [[ "$FAST_DEV_RUN" -eq 1 ]]; then
    TTT_EXPORT+=",FAST_DEV_RUN=1"
  fi
  TTT_ARGS=(--array "0-23%${ARRAY_CONCURRENCY}" --export "$TTT_EXPORT")
  if [[ -n "$LAST_DEP" ]]; then
    TTT_ARGS+=(--dependency "afterok:${LAST_DEP}")
  fi
  echo "Submitting TTT array job..."
  submit_job TTT_JOBID "${TTT_ARGS[@]}" hpc_ttt.sbatch
  LAST_DEP="$TTT_JOBID"
  echo "  ttt array job id: $TTT_JOBID (0-23%${ARRAY_CONCURRENCY})"
else
  echo "Skipping TTT submission"
fi

if [[ "$SKIP_ENSEMBLE" -eq 0 ]]; then
  ENS_EXPORT="ALL"
  if [[ "$PILLAR1_ONLY" -eq 1 ]]; then
    ENS_EXPORT+=",PILLAR1_ONLY=1"
  fi
  if [[ "$OVERWRITE_ENSEMBLE" -eq 1 ]]; then
    ENS_EXPORT+=",OVERWRITE=1"
  fi
  if [[ "$FORCE_CONSOLIDATE_TTT" -eq 1 ]]; then
    ENS_EXPORT+=",FORCE_CONSOLIDATE_TTT=1"
  fi
  ENS_ARGS=(--export "$ENS_EXPORT")
  if [[ -n "$LAST_DEP" ]]; then
    ENS_ARGS+=(--dependency "afterok:${LAST_DEP}")
  fi
  echo "Submitting final ensemble job..."
  submit_job ENSEMBLE_JOBID "${ENS_ARGS[@]}" hpc_full_pipeline.sbatch
  echo "  ensemble job id: $ENSEMBLE_JOBID"
else
  echo "Skipping ensemble submission"
fi

echo
echo "Submission summary:"
[[ -n "$PRETRAIN_JOBID" ]] && echo "  pretrain : $PRETRAIN_JOBID   (tail -f logs/slurm_pretrain_${PRETRAIN_JOBID}.log)"
[[ -n "$TTT_JOBID" ]] && echo "  ttt array: $TTT_JOBID        (tail -f logs/slurm_ttt_${TTT_JOBID}_<task>.log)"
[[ -n "$ENSEMBLE_JOBID" ]] && echo "  ensemble : $ENSEMBLE_JOBID   (tail -f logs/slurm_full_pipeline_${ENSEMBLE_JOBID}.log)"
echo "Monitor queue: squeue -u $USER"
