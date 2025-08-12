#!/bin/bash
# Merge per-chunk CSV/JSONL outputs into consolidated files

set -euo pipefail

CHUNKS_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunks-dir) CHUNKS_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$CHUNKS_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "--chunks-dir and --output-dir required" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Merge CSVs by taking header from the first and appending the rest without headers
OUT_CSV="$OUTPUT_DIR/ensemble_stats_merged.csv"
CSV_PARTS=( $(ls -1 ${CHUNKS_DIR}/ensemble_stats_chunk_*.csv 2>/dev/null || true) )
if [[ ${#CSV_PARTS[@]} -gt 0 ]]; then
  echo "Merging ${#CSV_PARTS[@]} CSV parts -> $OUT_CSV"
  head -n 1 "${CSV_PARTS[0]}" > "$OUT_CSV"
  for f in "${CSV_PARTS[@]}"; do
    tail -n +2 "$f" >> "$OUT_CSV"
  done
  echo "Merged CSV: $OUT_CSV"
else
  echo "No CSV parts found in $CHUNKS_DIR"
fi

# Merge JSONLs by concatenation
OUT_JSONL="$OUTPUT_DIR/ensemble_stats_merged.jsonl"
JSONL_PARTS=( $(ls -1 ${CHUNKS_DIR}/ensemble_stats_chunk_*.jsonl 2>/dev/null || true) )
if [[ ${#JSONL_PARTS[@]} -gt 0 ]]; then
  echo "Merging ${#JSONL_PARTS[@]} JSONL parts -> $OUT_JSONL"
  : > "$OUT_JSONL"
  for f in "${JSONL_PARTS[@]}"; do
    cat "$f" >> "$OUT_JSONL"
  done
  echo "Merged JSONL: $OUT_JSONL"
else
  echo "No JSONL parts found in $CHUNKS_DIR"
fi


