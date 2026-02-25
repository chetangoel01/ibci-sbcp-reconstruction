#!/bin/bash
# Submit a pillar1-only smoothing sigma sweep as sequential full-pipeline jobs.
# Jobs are chained to avoid races on submission.csv and shared output files.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/${USER}/ibci}"
PARTITION="${PARTITION:-}"
TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
SIGMAS_CSV="${SIGMAS_CSV:-0,0.25,0.5,0.75}"
USE_GPU="${USE_GPU:-0}"
OVERWRITE="${OVERWRITE:-0}"
SWEEP_TAG="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"

JOB_IDS=()
LAST_DEP=""

usage() {
  cat <<'EOF'
Usage: hpc_submit_sigma_sweep.sh [options]

Submit a sequence of `hpc_full_pipeline.sbatch` jobs for `--pillar1_only`
with different `SMOOTH_SIGMA` values. Each job saves a snapshot to
`submissions/submission_<tag>_pillar1_sigma_<sigma>.csv`.

Options:
  --repo-dir PATH         Repo root on HPC (default: /scratch/$USER/ibci)
  --partition NAME        Optional SLURM partition override
  --time HH:MM:SS         Walltime per job (default: 00:45:00)
  --sigmas CSV            Comma-separated sigma values (default: 0,0.25,0.5,0.75)
  --gpu                   Request `--gres=gpu:1` for each job
  --overwrite             Pass OVERWRITE=1 to full-pipeline jobs
  --tag NAME              Tag embedded in saved submission filenames
  -h, --help              Show help

Examples:
  ./hpc_submit_sigma_sweep.sh
  ./hpc_submit_sigma_sweep.sh --sigmas 0,0.1,0.2,0.3 --time 00:30:00
  ./hpc_submit_sigma_sweep.sh --gpu --partition l40s_publ
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --partition)
      PARTITION="$2"; shift 2 ;;
    --time)
      TIME_LIMIT="$2"; shift 2 ;;
    --sigmas)
      SIGMAS_CSV="$2"; shift 2 ;;
    --gpu)
      USE_GPU=1; shift ;;
    --overwrite)
      OVERWRITE=1; shift ;;
    --tag)
      SWEEP_TAG="$2"; shift 2 ;;
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
mkdir -p logs submissions

COMMON_SBAT=(sbatch --parsable --time "$TIME_LIMIT")
if [[ -n "$PARTITION" ]]; then
  COMMON_SBAT+=(--partition "$PARTITION")
fi
if [[ "$USE_GPU" -eq 1 ]]; then
  COMMON_SBAT+=(--gres gpu:1)
fi

IFS=',' read -r -a SIGMAS <<< "$SIGMAS_CSV"
if [[ "${#SIGMAS[@]}" -eq 0 ]]; then
  echo "No sigma values provided" >&2
  exit 1
fi

for raw_sigma in "${SIGMAS[@]}"; do
  sigma="$(echo "$raw_sigma" | xargs)"
  if [[ -z "$sigma" ]]; then
    continue
  fi
  safe_sigma="${sigma//./p}"
  safe_sigma="${safe_sigma//-/_neg_}"

  EXPORTS="ALL,PILLAR1_ONLY=1,OVERWRITE=${OVERWRITE},SMOOTH_SIGMA=${sigma}"
  EXPORTS+=",SAVE_SUBMISSION_AS=submission_${SWEEP_TAG}_pillar1_sigma_${safe_sigma}.csv"

  SBAT_ARGS=(--export "$EXPORTS")
  if [[ -n "$LAST_DEP" ]]; then
    SBAT_ARGS+=(--dependency "afterok:${LAST_DEP}")
  fi

  echo "Submitting sigma=${sigma} ..."
  jobid="$("${COMMON_SBAT[@]}" "${SBAT_ARGS[@]}" hpc_full_pipeline.sbatch)"
  jobid="${jobid%%;*}"
  if [[ -z "$jobid" ]]; then
    echo "Failed to parse job id for sigma=${sigma}" >&2
    exit 1
  fi

  JOB_IDS+=("${jobid}:${sigma}:${safe_sigma}")
  LAST_DEP="$jobid"
  echo "  job id: $jobid"
done

echo
echo "Submitted sigma sweep (sequential):"
for rec in "${JOB_IDS[@]}"; do
  IFS=':' read -r jobid sigma safe_sigma <<< "$rec"
  echo "  sigma=${sigma} -> job ${jobid}  (log: logs/slurm_full_pipeline_${jobid}.log)"
  echo "    output: submissions/submission_${SWEEP_TAG}_pillar1_sigma_${safe_sigma}.csv"
done
echo
echo "Monitor queue: squeue -u $USER"
