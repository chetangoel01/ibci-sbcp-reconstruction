#!/bin/bash
# Sync the local repo code to NYU Torch without copying large datasets.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-torch-dtn}"
REMOTE_DIR="${REMOTE_DIR:-/scratch/cg4652/ibci}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<'EOF'
Usage: hpc_phase2_sync.sh [options]

Options:
  --remote-host HOST   SSH host alias for rsync (default: torch-dtn)
  --remote-dir PATH    Remote repo root (default: /scratch/cg4652/ibci)
  -h, --help           Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-host)
      REMOTE_HOST="$2"; shift 2 ;;
    --remote-dir)
      REMOTE_DIR="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

rsync -avz \
  --exclude '.git/' \
  --exclude '.claude/' \
  --exclude '__pycache__/' \
  --exclude 'train/' \
  --exclude 'test/' \
  --exclude 'checkpoints/' \
  --exclude 'logs/' \
  --exclude 'results/' \
  --exclude 'phase2_outputs/' \
  --exclude '*.pyc' \
  "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Synced code to ${REMOTE_HOST}:${REMOTE_DIR}"
