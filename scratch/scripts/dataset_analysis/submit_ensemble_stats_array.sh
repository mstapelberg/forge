#!/bin/bash
# Orchestrate splitting, array submission, and merging for ensemble stats

set -euo pipefail

# ------------------------- User-configurable inputs ------------------------- #
DATA_DIR=""            # Directory containing train/val/test .xyz files
NUM_SPLITS=12           # Number of chunks / array size
SPLITS_JSON=""         # Path to structure_splits.json (optional)
MODEL_GLOB="/home/myless/Packages/forge/scratch/data/potentials/allegro/gen-8-aa/*.nequip.zip"
DEVICE="cuda"

# Cluster shape
NODES=3                 # Up to 5 per your note
TASKS_PER_NODE=4        # One per GPU (controls array throttle)
PARTITION="regular"
TIME_LIMIT="2-00:00:00"

# Output locations
BASE_OUT_DIR="/home/myless/Packages/forge/scratch/data/ensemble_stats_runs"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$BASE_OUT_DIR/run_${RUN_TAG}"
SPLIT_DIR="$RUN_DIR/splits_${NUM_SPLITS}"
CHUNK_OUT_DIR="$RUN_DIR/chunk_outputs"
MERGED_OUT_DIR="$RUN_DIR/merged"

usage() {
  cat <<EOF
Usage: $0 --data-dir /abs/path/to/data [--num-splits 12] [--splits-json /abs/path/structure_splits.json] \
          [--nodes 3] [--partition regular] [--time 2-00:00:00] [--device cuda]

Creates chunks, submits an array job with one task per chunk, and schedules a
merge job that depends on the array completion.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2;;
    --num-splits) NUM_SPLITS="$2"; shift 2;;
    --splits-json) SPLITS_JSON="$2"; shift 2;;
    --nodes) NODES="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --time) TIME_LIMIT="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --help|-h) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "--data-dir is required" >&2
  exit 1
fi

mkdir -p "$SPLIT_DIR" "$CHUNK_OUT_DIR" "$MERGED_OUT_DIR"

echo "Splitting .xyz files from: $DATA_DIR into $NUM_SPLITS chunks at $SPLIT_DIR"
python /home/myless/Packages/forge/scratch/scripts/dataset_analysis/split_xyz_chunks.py \
  --data-dir "$DATA_DIR" \
  --num-splits "$NUM_SPLITS" \
  --output-dir "$SPLIT_DIR"

ARRAY_RANGE="0-$((${NUM_SPLITS}-1))"

echo "Submitting array: $ARRAY_RANGE"
ARRAY_JOB_ID=$(sbatch \
  --parsable \
  --array="${ARRAY_RANGE}%$((NODES*TASKS_PER_NODE))" \
  -p "$PARTITION" \
  --time "$TIME_LIMIT" \
  /home/myless/Packages/forge/scratch/scripts/dataset_analysis/run_computed_ensemble_stats_array.sh \
    --model-glob "$MODEL_GLOB" \
    --splits-json "${SPLITS_JSON}" \
    --chunks-dir "$SPLIT_DIR" \
    --output-dir "$CHUNK_OUT_DIR" \
    --device "$DEVICE")

echo "Array job submitted: $ARRAY_JOB_ID"

echo "Submitting merge job dependent on array completion"
MERGE_JOB_ID=$(sbatch \
  --parsable \
  --dependency=afterok:${ARRAY_JOB_ID} \
  -p "$PARTITION" \
  --time 02:00:00 \
  /home/myless/Packages/forge/scratch/scripts/dataset_analysis/merge_ensemble_stats_outputs.sh \
    --chunks-dir "$CHUNK_OUT_DIR" \
    --output-dir "$MERGED_OUT_DIR")

echo "Merge job submitted: $MERGE_JOB_ID"
echo "Run directory: $RUN_DIR"


