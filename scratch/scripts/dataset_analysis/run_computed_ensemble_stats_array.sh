#!/bin/bash
#SBATCH --job-name=ensemble_stats
#SBATCH --output=ensemble_stats_%A_%a.out
#SBATCH --error=ensemble_stats_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -p regular
# Note: no constraint to allow any GPU node

# Usage:
#   sbatch --array=0-(N-1) run_computed_ensemble_stats_array.sh \
#     --model-glob "/abs/path/to/models/*.nequip.zip" \
#     --splits-json /abs/path/to/structure_splits.json \
#     --chunks-dir /abs/path/to/splits_dir \
#     --output-dir /abs/path/to/output_dir \
#     --device cuda
#
# Submit with up to M nodes and 4 tasks/node:
#   sbatch --nodes M --ntasks-per-node 4 --array=0-(N-1) run_computed_ensemble_stats_array.sh ...

set -euo pipefail

# ------------------------- Parse lightweight args -------------------------- #
MODEL_GLOB=""
SPLITS_JSON=""
CHUNKS_DIR=""
OUTPUT_DIR=""
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-glob)
      MODEL_GLOB="$2"; shift 2;;
    --splits-json)
      SPLITS_JSON="$2"; shift 2;;
    --chunks-dir)
      CHUNKS_DIR="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    *)
      echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

if [[ -z "$CHUNKS_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "--chunks-dir and --output-dir are required" >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Load Conda environment
source /home/myless/.mambaforge/etc/profile.d/conda.sh
conda activate forge-allegro

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

CHUNK_FILE=$(printf "%s/chunk_%04d.xyz" "$CHUNKS_DIR" "$TASK_ID")
if [[ ! -f "$CHUNK_FILE" ]]; then
  echo "Chunk not found: $CHUNK_FILE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

OUT_CSV="$OUTPUT_DIR/ensemble_stats_chunk_${TASK_ID}.csv"
OUT_JSONL="$OUTPUT_DIR/ensemble_stats_chunk_${TASK_ID}.jsonl"

srun --ntasks=1 --gres=gpu:1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-8} \
  python /home/myless/Packages/forge/scratch/scripts/dataset_analysis/computed_ensemble_stats.py \
  --model-glob "${MODEL_GLOB}" \
  --data-paths "$CHUNK_FILE" \
  --device "$DEVICE" \
  --out-csv "$OUT_CSV" \
  --out-jsonl "$OUT_JSONL" \
  ${SPLITS_JSON:+--splits-json "$SPLITS_JSON"}

echo "Wrote: $OUT_CSV"


