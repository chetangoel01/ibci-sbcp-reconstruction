#!/bin/bash
# Clean generated files for the iBCI pipeline on HPC.
# Default mode targets stale TTT outputs/logs while keeping pretrained checkpoints.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/${USER}/ibci}"
MODE="ttt"
DRY_RUN=1
KEEP_PRETRAIN=1
KEEP_GAUSSIAN=1
KEEP_SUBMISSION=1

usage() {
  cat <<'EOF'
Usage: hpc_cleanup_outputs.sh [options]

Options:
  --repo-dir PATH      Repository root (default: /scratch/$USER/ibci)
  --ttt-only           Clean only TTT outputs/logs (default)
  --full               Clean most generated artifacts (results, logs, slurm logs, checkpoints)
  --keep-pretrained    Keep checkpoints/mae_pretrained.pt (default)
  --remove-pretrained  Remove checkpoints/mae_pretrained.pt in --full mode
  --keep-gaussian      Keep results/gaussian_predictions.csv (default)
  --remove-gaussian    Remove results/gaussian_predictions.csv in --full mode
  --keep-submission    Keep submission.csv (default)
  --remove-submission  Remove submission.csv in --full mode
  --yes                Execute deletions (default is dry-run)
  --dry-run            Print what would be deleted (default)
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --ttt-only)
      MODE="ttt"; shift ;;
    --full)
      MODE="full"; shift ;;
    --keep-pretrained)
      KEEP_PRETRAIN=1; shift ;;
    --remove-pretrained)
      KEEP_PRETRAIN=0; shift ;;
    --keep-gaussian)
      KEEP_GAUSSIAN=1; shift ;;
    --remove-gaussian)
      KEEP_GAUSSIAN=0; shift ;;
    --keep-submission)
      KEEP_SUBMISSION=1; shift ;;
    --remove-submission)
      KEEP_SUBMISSION=0; shift ;;
    --yes)
      DRY_RUN=0; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
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

declare -a TARGETS=()

# Always safe to clear stale TTT outputs for a fresh TTT rerun.
TARGETS+=(
  "$REPO_DIR/results/ttt_predictions/*.csv"
  "$REPO_DIR/results/mae_predictions.csv"
  "$REPO_DIR/checkpoints/mae_ttt_*.pt"
  "$REPO_DIR/slurm_ttt_*.out"
  "$REPO_DIR/slurm_ttt_*.err"
  "$REPO_DIR/logs/slurm_ttt_*.log"
  "$REPO_DIR/logs/pillar2_mae_ttt.log"
)

if [[ "$MODE" == "full" ]]; then
  TARGETS+=(
    "$REPO_DIR/slurm_pretrain_*.out"
    "$REPO_DIR/slurm_pretrain_*.err"
    "$REPO_DIR/slurm_full_pipeline_*.out"
    "$REPO_DIR/slurm_full_pipeline_*.err"
    "$REPO_DIR/logs/slurm_pretrain_*.log"
    "$REPO_DIR/logs/slurm_full_pipeline_*.log"
    "$REPO_DIR/logs/pillar2_mae_train.log"
    "$REPO_DIR/logs/ensemble.log"
    "$REPO_DIR/logs/consolidate_ttt_predictions.log"
    "$REPO_DIR/results/ensemble_weights.json"
    "$REPO_DIR/results/ensemble_session_summary.csv"
    "$REPO_DIR/results/mae_cv.csv"
    "$REPO_DIR/results/gaussian_cv.csv"
    "$REPO_DIR/results/analysis_by_difficulty.csv"
    "$REPO_DIR/results/analysis_by_channel.csv"
  )
  if [[ "$KEEP_GAUSSIAN" -eq 0 ]]; then
    TARGETS+=("$REPO_DIR/results/gaussian_predictions.csv")
  fi
  if [[ "$KEEP_SUBMISSION" -eq 0 ]]; then
    TARGETS+=("$REPO_DIR/submission.csv")
  fi
  if [[ "$KEEP_PRETRAIN" -eq 0 ]]; then
    TARGETS+=("$REPO_DIR/checkpoints/mae_pretrained.pt")
    TARGETS+=("$REPO_DIR/checkpoints/pretrain_epochs/*.pt")
  fi
fi

print_matches() {
  local pattern
  local found=0
  shopt -s nullglob
  for pattern in "${TARGETS[@]}"; do
    local matches=( $pattern )
    if [[ ${#matches[@]} -eq 0 ]]; then
      continue
    fi
    for path in "${matches[@]}"; do
      found=1
      printf '%s\n' "$path"
    done
  done
  shopt -u nullglob
  return $found
}

MATCHES=$(print_matches || true)
if [[ -z "$MATCHES" ]]; then
  echo "No matching files to clean under $REPO_DIR"
  exit 0
fi

echo "Cleanup mode: $MODE"
echo "Repo dir: $REPO_DIR"
echo "Dry run: $DRY_RUN"
echo
printf '%s\n' "$MATCHES"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "Dry-run only. Re-run with --yes to delete these files."
  exit 0
fi

while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  rm -f -- "$path"
done <<< "$MATCHES"

echo "Cleanup complete."
